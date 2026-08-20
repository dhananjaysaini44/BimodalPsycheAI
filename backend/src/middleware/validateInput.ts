import { Request, Response, NextFunction } from "express";

export default function validateInput(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Skeleton validation middleware
  next();
}
