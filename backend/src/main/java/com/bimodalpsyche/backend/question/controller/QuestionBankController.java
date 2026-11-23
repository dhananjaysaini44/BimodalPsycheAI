package com.bimodalpsyche.backend.question.controller;

import com.bimodalpsyche.backend.question.service.QuestionBankService;
import com.bimodalpsyche.backend.question.service.QuestionBankService.QuestionBankResponse;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/questions")
public class QuestionBankController {
    private final QuestionBankService questionBankService;

    public QuestionBankController(QuestionBankService questionBankService) {
        this.questionBankService = questionBankService;
    }

    @GetMapping("/active")
    public QuestionBankResponse getActiveBank() {
        return questionBankService.getActiveBank();
    }
}
