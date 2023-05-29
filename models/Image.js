import {Sequelize} from "sequelize";
import db from "../config/Database.js";

const { DataTypes } = Sequelize;

const Image = db.define('image', {
  name: {
    type: DataTypes.STRING
  },
  predict:{
    type:DataTypes.STRING
  }
}, {
  freezeTableName: true
});

export default Image;
