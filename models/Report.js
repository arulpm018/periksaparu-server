import {Sequelize} from "sequelize";
import db from "../config/Database.js";
import Image from "./Image.js";


const {DataTypes} = Sequelize;

const Report= db.define('report',{
    patientname:{
        type:DataTypes.STRING
    },
    result:{
        type:DataTypes.STRING
    },
    userId:{
        type:DataTypes.INTEGER
    },
    analisisdokter:{
        type:DataTypes.STRING
    },
    imageid:{
        type:DataTypes.INTEGER
    }

    
},{
    freezeTableName:true
});

Report.belongsTo(Image, { foreignKey: 'imageid' });
export default Report;