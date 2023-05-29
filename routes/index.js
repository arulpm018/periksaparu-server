import express from "express";
import {getUsers,Register,Login,Logout,Upload,upload, getResult, getReport, getDoctor, getDistinctPatientNames, updateAnalisisDokter, getSpesificReport, getSpesificDistinctPatientNames} from "../controllers/Users.js";
import { verifyToken } from "../middleware/VerifyToken.js";
import { refreshToken } from "../controllers/RefreshToken.js";
const router = express.Router();




router.get('/users',getUsers);
router.put('/report/:id',updateAnalisisDokter)
router.get('/report',getReport);
router.get('/spesificreport',getSpesificReport);
router.get('/patientname',getDistinctPatientNames);
router.get('/spesificpatientname',getSpesificDistinctPatientNames);
router.get('/dokter',getDoctor)
router.post('/users',Register);
router.post('/login',Login);
router.get('/token',refreshToken);
router.delete('/logout',Logout);
router.post('/upload',upload.single('image'),Upload)
router.get('/result',verifyToken,getResult);






export default router;
