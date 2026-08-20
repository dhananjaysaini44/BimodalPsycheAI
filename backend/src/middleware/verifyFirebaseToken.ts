import { Request, Response, NextFunction } from "express";
import * as admin from "firebase-admin";

const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_PATH || "";

if (serviceAccountPath && !admin.apps.length) {
  try {
    const serviceAccount = require(serviceAccountPath);
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
  } catch (error) {
    console.warn("Could not initialize Firebase admin with service account:", error);
  }
}

export interface AuthenticatedRequest extends Request {
  userId?: string;
}

export default async function verifyFirebaseToken(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<any> {
  const authHeader = req.headers.authorization || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : null;

  if (!token) return res.status(401).json({ error: "Missing auth token" });

  try {
    const decoded = await admin.auth().verifyIdToken(token);
    req.userId = decoded.uid;
    next();
  } catch (err) {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}
