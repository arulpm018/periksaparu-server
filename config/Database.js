import {Sequelize} from "sequelize";

const db = new Sequelize('periksaparu','root','',{
    host:"localhost",
    dialect:"mysql"
});

export default db;