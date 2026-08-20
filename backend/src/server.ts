import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";

import authRoutes from "./routes/auth.routes";
import sessionRoutes from "./routes/session.routes";
import historyRoutes from "./routes/history.routes";
import errorHandler from "./middleware/errorHandler";

const app = express();
app.use(cors());
app.use(express.json());

app.use("/auth", authRoutes);
app.use("/session", sessionRoutes);
app.use("/history", historyRoutes);

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.use(errorHandler);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
