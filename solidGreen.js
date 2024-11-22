// import express from 'express';
// import cors from 'cors';
// const app = express();



// app.use(express.static('public'))
// app.use(express.json())
// app.use(cors())




// await db.migrate();



// app.post('/api/predict', async (req, res) => {
//     const { make,model, vehicle, engineSize,cylinders, transmission,fuelType, fuelConsumptionCity,fuelConsumptionHwy,fuelConsumptionComb,fuelConsumptionCombMpg,cO2Emissions} = req.body
//     const emissions = await db.run(`INSERT INTO PREDICTION (make,model, vehicle, engineSize,cylinders, transmission,fuelType, fuelConsumptionCity,fuelConsumptionHwy,fuelConsumptionComb,fuelConsumptionCombMpg,cO2Emission) VALUES (?,?,?,?,?)`, [make,model, vehicle, engineSize,cylinders, transmission,fuelType, fuelConsumptionCity,fuelConsumptionHwy,fuelConsumptionComb,fuelConsumptionCombMpg,cO2Emissions]);
//     res.status(200).json({ message: 'Emissions Predicted' });
// })

// const PORT = process.env.PORT || 4000
// app.listen(PORT, () => console.log(`Server started ${PORT}`))
