import {Sequelize} from "sequelize";
import db from "../config/Database.js";

import Image from './Image.js';

import Report from './Report.js'

const {DataTypes} = Sequelize;

const Users = db.define('users',{
    name:{
        type:DataTypes.STRING
    },
    email:{
        type:DataTypes.STRING
    },
    password:{
        type:DataTypes.STRING
    },
    role:{
        type:DataTypes.STRING
    },
    refresh_token:{
        type:DataTypes.TEXT
    }
    
},{
    freezeTableName:true
});

Report.belongsTo(Users);
Image.belongsTo(Users);

export default Users;