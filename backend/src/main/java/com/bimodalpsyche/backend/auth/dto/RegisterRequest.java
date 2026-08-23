package com.bimodalpsyche.backend.auth.dto;

import com.bimodalpsyche.backend.user.model.EmploymentType;
import com.bimodalpsyche.backend.user.model.Gender;
import com.bimodalpsyche.backend.user.model.MaritalStatus;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Getter;

import java.time.LocalDate;

@Getter
public class RegisterRequest {

    @NotBlank(message = "Email cannot be blank")
    @Email(message = "Email must be valid")
    private String email;

    @NotBlank(message = "Password cannot be blank")
    @Size(min = 8, max = 100, message = "Password must be between 8 and 100 characters")
    private String password;

    @NotBlank(message = "First name cannot be blank")
    private String firstName;

    private String lastName;

    @NotNull(message = "Gender is required")
    private Gender gender;

    @NotNull(message = "Marital status is required")
    private MaritalStatus maritalStatus;

    private String jobTitle;

    @NotNull(message = "Employment type is required")
    private EmploymentType employmentType;

    private String phoneNumber;

    private Integer workingHoursPerWeek;

    @NotNull(message = "Date of birth is required")
    private LocalDate dateOfBirth;
}