import { Router, Response } from "express";
import verifyFirebaseToken, { AuthenticatedRequest } from "../middleware/verifyFirebaseToken";

const router = Router();

// /session/start
router.post("/start", verifyFirebaseToken, (req: AuthenticatedRequest, res: Response) => {
  res.json({ message: "Start session skeleton", userId: req.userId });
});

// /session/turn
router.post("/turn", verifyFirebaseToken, (req: AuthenticatedRequest, res: Response) => {
  res.json({ message: "Submit turn skeleton", userId: req.userId });
});

// /session/complete
router.post("/complete", verifyFirebaseToken, (req: AuthenticatedRequest, res: Response) => {
  res.json({ message: "Complete session skeleton", userId: req.userId });
});

export default router;
