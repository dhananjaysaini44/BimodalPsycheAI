import { Router, Request, Response } from "express";

const router = Router();

router.post("/register", (req: Request, res: Response) => {
  res.json({ message: "Auth register skeleton" });
});

router.post("/login", (req: Request, res: Response) => {
  res.json({ message: "Auth login skeleton" });
});

export default router;
