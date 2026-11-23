package com.bimodalpsyche.backend.auth.dto;

import com.bimodalpsyche.backend.user.model.EmploymentType;
import com.bimodalpsyche.backend.user.model.Gender;
import com.bimodalpsyche.backend.user.model.MaritalStatus;
import jakarta.validation.constraints.*;
import lombok.Getter;

import java.time.LocalDate;
import java.util.Locale;

@Getter
public class RegisterRequest {

    @NotBlank(message = "Email cannot be blank")
    @Email(message = "Email must be valid")
    @Size(max = 255, message = "Email is too long")
    private String email;

    // BCrypt only uses the first 72 bytes of a password
    @NotBlank(message = "Password cannot be blank")
    @Size(min = 8, max = 72, message = "Password must be between 8 and 72 characters")
    private String password;

    @NotBlank(message = "First name cannot be blank")
    @Size(max = 100, message = "First name must be at most 100 characters")
    private String firstName;

    @Size(max = 100, message = "Last name must be at most 100 characters")
    private String lastName;

    @NotNull(message = "Gender is required")
    private Gender gender;

    @NotNull(message = "Marital status is required")
    private MaritalStatus maritalStatus;

    @Size(max = 100, message = "Job title must be at most 100 characters")
    private String jobTitle;

    @NotNull(message = "Employment type is required")
    private EmploymentType employmentType;

    @Pattern(regexp = "^\\+?[0-9]{7,15}$", message = "Phone number must be 7-15 digits, optionally starting with +")
    private String phoneNumber;

    @Min(value = 0, message = "Working hours cannot be negative")
    @Max(value = 168, message = "Working hours cannot exceed 168 per week")
    private Integer workingHoursPerWeek;

    @NotNull(message = "Date of birth is required")
    @Past(message = "Date of birth must be in the past")
    private LocalDate dateOfBirth;

    @AssertTrue(message = "Terms of Service must be accepted")
    private boolean termsAccepted;

    @AssertTrue(message = "Privacy Policy must be accepted")
    private boolean privacyPolicyAccepted;

    public String getEmail() {
        return email == null ? null : email.trim().toLowerCase(Locale.ROOT);
    }
}
