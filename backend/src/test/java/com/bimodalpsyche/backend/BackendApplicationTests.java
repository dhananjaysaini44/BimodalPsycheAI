package com.bimodalpsyche.backend;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest(properties = "app.encryption.key=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
@AutoConfigureMockMvc
class BackendApplicationTests {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void contextLoads() {
    }

    @Test
    void sessionDataRequiresAuthentication() throws Exception {
        mockMvc.perform(get("/session/draft"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void unknownAuthRoutesAreNotPublic() throws Exception {
        mockMvc.perform(get("/auth/private-data"))
                .andExpect(status().isUnauthorized());
    }

}
