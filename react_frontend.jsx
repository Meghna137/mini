import { useState } from "react";
import axios from "axios";

export default function PredictorApp() {
    const [features, setFeatures] = useState([13.85, 26.34, 49.51, 58.42, 64.87, 57.60, 46.75, 43.32, 12.75, 39.01, 18, 45.17]);
    const [model, setModel] = useState("RandomForest");
    const [prediction, setPrediction] = useState(null);
    const models = ["RandomForest", "XGBoost", "NeuralNetwork"];

    const handlePredict = async () => {
        try {
            const response = await axios.post("http://127.0.0.1:5000/predict", {
                model,
                features
            });
            setPrediction(response.data.predicted_traits);
        } catch (error) {
            console.error("Prediction Error", error);
        }
    };

    return (
        <div className="p-6 max-w-lg mx-auto bg-white rounded-xl shadow-md">
            <h2 className="text-xl font-bold mb-4">Plant Trait Predictor</h2>
            <label className="block mb-2">Select Model:</label>
            <select 
                className="p-2 border rounded mb-4"
                value={model} 
                onChange={(e) => setModel(e.target.value)}>
                {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>

            <div className="mb-4">
                <label className="block mb-2">Input Features:</label>
                {features.map((f, i) => (
                    <input 
                        key={i} 
                        type="number" 
                        value={f} 
                        className="p-2 border rounded mb-2 w-full"
                        onChange={(e) => {
                            const newFeatures = [...features];
                            newFeatures[i] = parseFloat(e.target.value);
                            setFeatures(newFeatures);
                        }}
                    />
                ))}
            </div>

            <button 
                className="bg-blue-500 text-white p-2 rounded w-full"
                onClick={handlePredict}>
                Predict
            </button>

            {prediction && (
                <div className="mt-4 p-4 bg-gray-100 rounded">
                    <h3 className="font-bold">Predicted Traits:</h3>
                    <ul>
                        {prediction.map((p, i) => <li key={i}>{p.toFixed(2)}</li>)}
                    </ul>
                </div>
            )}
        </div>
    );
}
