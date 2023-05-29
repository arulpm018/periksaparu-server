import Users from "../models/UserModel.js";
import Image from "../models/Image.js";
import Report from "../models/Report.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import cookieParser from "cookie-parser";
import express from "express";
import multer from "multer";
import {Sequelize} from "sequelize";





const app = express();
app.use(cookieParser());

export const getUsers = async(req,res) => {
    try {
        const userId = req.cookies.userId;
        const users = await Users.findAll({
            where: {
                id: userId
            },
            attributes:['id','name','email','role']
        });
        res.json(users);    
    } catch (error) {
        console.log(error);
    }
}

export const Register = async(req,res) =>{
    const {name,email,password,confPassword,role} = req.body;
    console.log(role);
    if(password!==confPassword) return res.status(400).json({msg:"Password dan Confirm Password tidak cocok"});
    const salt = await bcrypt.genSalt();
    const hashPassword = await bcrypt.hash(password,salt);
    try {
        await Users.create({
            name:name,
            email:email,
            password:hashPassword,
            role:role

        });
        res.json({msg:"Register Berhasil"});
    } catch (error) {
        console.log(error); 
    }
}

export const Login = async(req,res)=>{
    try {
        const user = await Users.findAll({
            where:{
                email:req.body.email
            }
        })
        const match = await bcrypt.compare(req.body.password,user[0].password);
        if(!match) return res.status(400).json({msg:"password anda salah"});
        const userId = user[0].id;
        const name = user[0].name;
        const email = user[0].email;
        const role = user[0].role;
        const accessToken = jwt.sign({userId,name,email,role},process.env.ACCESS_TOKEN_SECRET,{
            expiresIn:'20s'
        });
        const refreshToken = jwt.sign({userId,name,email,role},process.env.REFRESH_TOKEN_SECRET,{
            expiresIn:'1d'
        });
        await Users.update({refresh_token:refreshToken},{
            where:{
                id:userId
            }
        });
        res.cookie('userId', userId, {
            httpOnly: true,
            maxAge: 24*60*60*1000 // waktu kadaluarsa cookie dalam milidetik
        });
        res.cookie('refreshToken',refreshToken,{
            httpOnly:true,
            maxAge:24*60*60*1000
        });
        res.json({accessToken});

    } catch (error) {
        res.status(404).json({msg:"Email tidak ditemukan"});
        
    }
}

export const Logout = async(req,res)=> {
    const refreshToken = req.cookies.refreshToken;
    if(!refreshToken) return res.sendStatus(204);
    const user = await Users.findAll({
        where:{
            refresh_token: refreshToken
        }
    });
    if(!user[0]) return res.sendstatus(204);
    const userId= user[0].id;
    await Users.update({refresh_token:null},{
        where:{
            id:userId
        }
    });
    res.clearCookie('refreshToken');
    return res.sendStatus(200);

}


const storage = multer.diskStorage({
    destination: function (req, file, cb) {
      cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
      cb(null, file.originalname);
    },
  });

export const upload = multer({ storage: storage });

function getRandomElement(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

export const Upload = async(req,res)=> {

    try {

        const userid = req.cookies.userId;
        const idDoctor = req.body.idDoctor;
        const imageName = req.file.filename;
        const patientName = req.body.patientname;


        const array =["normal","pneumonia","tuberkulosis"]

        const predict = getRandomElement(array)
        const image = await Image.create({
            name:imageName,
            userId:userid,
            predict: predict
        });

        const imageId = image.id

        await Report.create({
            patientname:patientName,
            result:predict,
            userId: idDoctor,
            analisisdokter:"-",
            imageid : imageId
        });

        res.cookie('imageName', imageName, {
            httpOnly: true,
            maxAge: 24*60*60*1000 // waktu kadaluarsa cookie dalam milidetik
        });
        res.json({msg:`Report berhasil dikirim`});
    } catch (error) {
        console.log(error)
    }

}

export const getResult = async(req,res) => {
    try {

        const imageName = req.cookies.imageName;
        const image = await Image.findAll({
            where: {
                name:imageName
            },
            attributes:['name','predict']
        });
        res.json(image);    
    } catch (error) {
        console.log(error);
    }
}


export const getReport= async(req,res) => {
    try {
        const report = await Report.findAll({
            attributes:['id','patientname','result','userId','analisisdokter','createdAt'],
            include: {
                model: Image,
                attributes: ['name'], // Specify the attributes you want to retrieve from the Image model
            },
        });
        res.json(report);    
    } catch (error) {
        console.log(error);
    }
}

export const getDistinctPatientNames = async (req, res) => {
    try {
      const patientNames = await Report.findAll({
        attributes: [
          [Sequelize.fn('DISTINCT', Sequelize.col('patientname')), 'patientname']
        ]
      });
  
      res.json(patientNames);
    } catch (error) {
      console.log(error);
      res.status(500).json({ error: 'An error occurred while retrieving patient names' });
    }
  };

  export const getSpesificDistinctPatientNames = async (req, res) => {
    try {
        const userid = req.cookies.userId;
      const patientNames = await Report.findAll({
        where: {
            userId:userid
        },
        attributes: [
          [Sequelize.fn('DISTINCT', Sequelize.col('patientname')), 'patientname']
        ]
      });
  
      res.json(patientNames);
    } catch (error) {
      console.log(error);
      res.status(500).json({ error: 'An error occurred while retrieving patient names' });
    }
  };

export const getDoctor= async(req,res) => {
    try {
        const userid = req.cookies.userId;
        const doctor= await Users.findAll({
            where: {
                role:"Dokter"
            },
            attributes:['name','id']
        });
        res.json(doctor);    
    } catch (error) {
        console.log(error);
    }
}



export const getSpesificReport= async(req,res) => {
    try {
        const userid = req.cookies.userId;
        const report = await Report.findAll({
            where: {
                userId:userid
            },
            attributes:['id','patientname','result','userId','analisisdokter','createdAt'],
            include: {
                model: Image,
                attributes: ['name'], // Specify the attributes you want to retrieve from the Image model
            },
        });
        res.json(report);    
    } catch (error) {
        console.log(error);
    }
}


export const updateAnalisisDokter = async (req, res) => {
    const { id } = req.params;
    const { analisisdokter } = req.body;
  
    try {

        await Report.update(
            {analisisdokter:analisisdokter},
            {
                where:{
                    id:id
            }
        });
      
        res.status(200).json({ message: 'Analisisdokter updated successfully' });
        } catch (error) {
        console.log(error);
        res.status(500).json({ error: 'Internal server error' });
        }
    };












