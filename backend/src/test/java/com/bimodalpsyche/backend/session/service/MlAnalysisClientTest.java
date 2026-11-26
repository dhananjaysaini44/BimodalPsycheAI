package com.bimodalpsyche.backend.session.service;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MlAnalysisClientTest {
    @Test
    void sendsTextAsExpectedByMlOrchestrator() throws IOException {
        List<String> requestHeaders = new ArrayList<>();
        List<String> requestBodies = new ArrayList<>();
        List<String> contentTypes = new ArrayList<>();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/orchestrator/sentiment", exchange -> {
            requestHeaders.add(exchange.getRequestHeaders().toString());
            requestBodies.add(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            contentTypes.add(exchange.getRequestHeaders().getFirst("Content-Type"));
            respond(exchange, "{\"sentiment_score\":-0.5106}");
        });
        server.createContext("/orchestrator/crisis", exchange -> {
            requestHeaders.add(exchange.getRequestHeaders().toString());
            requestBodies.add(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            contentTypes.add(exchange.getRequestHeaders().getFirst("Content-Type"));
            respond(exchange, "{\"is_crisis\":false}");
        });
        server.start();
        try {
            MlAnalysisClient client = new MlAnalysisClient("http://127.0.0.1:" + server.getAddress().getPort());

            MlAnalysisClient.TurnAnalysisResult result = client.analyze("I feel low today.");

            assertEquals(-0.5106f, result.sentimentScore());
            assertEquals(false, result.crisisDetected());
            assertEquals(2, requestBodies.size());
            assertTrue(requestBodies.stream().allMatch(body -> body.equals("{\"text\":\"I feel low today.\"}")));
            assertTrue(contentTypes.stream().allMatch(type -> type.startsWith("application/json")));
            assertTrue(requestHeaders.stream().noneMatch(headers -> headers.contains("Upgrade=[h2c]")));
        } finally {
            server.stop(0);
        }
    }

    private static void respond(com.sun.net.httpserver.HttpExchange exchange, String body) throws IOException {
        byte[] response = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, response.length);
        try (var output = exchange.getResponseBody()) {
            output.write(response);
        }
    }
}
