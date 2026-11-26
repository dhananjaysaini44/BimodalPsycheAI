package com.bimodalpsyche.backend.user.dto;

import com.bimodalpsyche.backend.user.model.EmploymentType;
import com.bimodalpsyche.backend.user.model.Gender;
import com.bimodalpsyche.backend.user.model.MaritalStatus;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Past;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;

import java.time.LocalDate;

@Getter
public class UpdateProfileRequest {
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
}
