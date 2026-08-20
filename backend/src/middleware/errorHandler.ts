import { Request, Response, NextFunction } from "express";

export default function errorHandler(
  err: any,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  console.error("Backend Error:", err);
  const status = err.status || 500;
  const message = err.message || "Internal Server Error";
  res.status(status).json({ error: message });
}
