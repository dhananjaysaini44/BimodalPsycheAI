import { Router, Response } from "express";
import verifyFirebaseToken, { AuthenticatedRequest } from "../middleware/verifyFirebaseToken";

const router = Router();

router.get("/", verifyFirebaseToken, (req: AuthenticatedRequest, res: Response) => {
  res.json({ message: "Get history skeleton", userId: req.userId });
});

export default router;
