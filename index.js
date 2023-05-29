import express from "express";
import db from "./config/Database.js";
import dotenv from "dotenv";
import router from "./routes/index.js";
import cookieParser from "cookie-parser";
import cors from "cors";
import Report from "./models/Report.js";
const app = express();
dotenv.config();

try{
    await db.authenticate();
    console.log('database connected...');
} catch(error){
    console.error(error);
}

app.use('/static', express.static('./uploads'));
app.use(cors({credentials:true,origin:'http://localhost:3000'}));
app.use(cookieParser());
app.use(express.json());
app.use(router);

app.listen(5000,()=> console.log('server running at port 5000'))
