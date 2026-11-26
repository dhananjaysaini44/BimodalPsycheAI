package com.bimodalpsyche.backend.session.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.bimodalpsyche.backend.assessment.model.Severity;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.HttpStatusCode;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.Map;

@Component
public class MlAnalysisClient {
    private final RestClient restClient;

    public MlAnalysisClient(@Value("${app.ml.base-url:http://localhost:8000}") String baseUrl) {
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(
                HttpClient.newBuilder()
                        .version(HttpClient.Version.HTTP_1_1)
                        .connectTimeout(Duration.ofSeconds(3))
                        .build());
        requestFactory.setReadTimeout(Duration.ofSeconds(5));
        this.restClient = RestClient.builder().baseUrl(baseUrl).requestFactory(requestFactory).build();
    }

    public TurnAnalysisResult analyze(String text) {
        SentimentResponse sentiment = postJson(
                "/orchestrator/sentiment", Map.of("text", text), SentimentResponse.class);
        CrisisResponse crisis = postJson(
                "/orchestrator/crisis", Map.of("text", text), CrisisResponse.class);

        if (sentiment == null || sentiment.sentimentScore() == null
                || !Float.isFinite(sentiment.sentimentScore())
                || sentiment.sentimentScore() < -1f || sentiment.sentimentScore() > 1f
                || crisis == null || crisis.crisisDetected() == null) {
            throw new IllegalStateException("ML analysis service returned an invalid response");
        }
        return new TurnAnalysisResult(sentiment.sentimentScore(), crisis.crisisDetected());
    }

    public ModelAnalysisResult analyzeVoice(byte[] audio, String mimeType, String filename) {
        ByteArrayResource audioResource = new ByteArrayResource(audio) {
            @Override
            public String getFilename() {
                return filename;
            }
        };
        HttpHeaders partHeaders = new HttpHeaders();
        partHeaders.setContentType(MediaType.parseMediaType(mimeType));
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("file", new HttpEntity<>(audioResource, partHeaders));
        return validated(restClient.post()
                .uri("/analyze/voice/")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .body(ModelAnalysisResult.class));
    }

    public ModelAnalysisResult analyzeText(String text) {
        return validated(postJson("/analyze/text/", Map.of("text", text), ModelAnalysisResult.class));
    }

    public ModelAnalysisResult fuse(ModelAnalysisResult text, ModelAnalysisResult voice) {
        return validated(postJson("/fusion/",
                new FusionRequest(text.score(), text.probability(), voice.score(), voice.probability()),
                ModelAnalysisResult.class));
    }

    private <T> T postJson(String endpoint, Object request, Class<T> responseType) {
        return restClient.post()
                .uri(endpoint)
                .contentType(MediaType.APPLICATION_JSON)
                .body(request)
                .retrieve()
                .onStatus(HttpStatusCode::isError, (httpRequest, response) -> {
                    throw new MlServiceHttpException(endpoint, response.getStatusCode().value());
                })
                .body(responseType);
    }

    public static class MlServiceHttpException extends RuntimeException {
        private final String endpoint;
        private final int statusCode;

        public MlServiceHttpException(String endpoint, int statusCode) {
            super("ML endpoint " + endpoint + " returned HTTP " + statusCode);
            this.endpoint = endpoint;
            this.statusCode = statusCode;
        }

        public String endpoint() {
            return endpoint;
        }

        public int statusCode() {
            return statusCode;
        }
    }

    private ModelAnalysisResult validated(ModelAnalysisResult result) {
        if (result == null || result.score() == null || !Float.isFinite(result.score())
                || result.probability() == null || !Float.isFinite(result.probability())
                || result.probability() < 0f || result.probability() > 1f
                || result.severity() == null) {
            throw new IllegalStateException("ML analysis service returned an invalid assessment result");
        }
        try {
            Severity.valueOf(result.severity());
        } catch (IllegalArgumentException ex) {
            throw new IllegalStateException("ML analysis service returned an invalid severity", ex);
        }
        return result;
    }

    public record TurnAnalysisResult(float sentimentScore, boolean crisisDetected) {
    }

    public record ModelAnalysisResult(Float score, Float probability, String severity, String explanation) {
        public Severity severityValue() {
            return Severity.valueOf(severity);
        }
    }

    private record FusionRequest(
            @JsonProperty("text_score") Float textScore,
            @JsonProperty("text_probability") Float textProbability,
            @JsonProperty("voice_score") Float voiceScore,
            @JsonProperty("voice_probability") Float voiceProbability) {
    }

    private record SentimentResponse(@JsonProperty("sentiment_score") Float sentimentScore) {
    }

    private record CrisisResponse(@JsonProperty("is_crisis") Boolean crisisDetected) {
    }
}
