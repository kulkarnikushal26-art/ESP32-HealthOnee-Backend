require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');
const nodemailer = require('nodemailer');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;
let db;

// 1. Initialize MongoDB Connection
const client = new MongoClient(process.env.MONGODB_URI);
async function connectDB() {
    try {
        await client.connect();
        db = client.db();
        console.log("Connected successfully to MongoDB");
    } catch (err) {
        console.error("MongoDB connection failed:", err);
        process.exit(1);
    }
}
connectDB();

// 2. Setup Email Transporter
const transporter = nodemailer.createTransport({
    service: process.env.EMAIL_SERVICE,
    auth: {
        user: process.env.EMAIL_USER,
        制造: process.env.EMAIL_PASS // Use an App Password if using Gmail
    }
});

// Default configurations if none exist in the database yet
const DEFAULT_CONFIGS = {
    heartRate: { min: 60, max: 100 },
    spo2: { min: 95, max: 100 },
    bodyTemp: { min: 36.0, max: 37.5 }
};

// 3. Telemetry Processing Endpoint
app.post('/api/telemetry', async (req, res) => {
    const { deviceID, heartRate, spo2, bodyTemp } = req.body;

    if (!deviceID || heartRate === undefined || spo2 === undefined || bodyTemp === undefined) {
        return res.status(400).json({ error: "Missing required fields within telemetry payload." });
    }

    try {
        const timestamp = new Date();
        const currentReadings = { heartRate, spo2, bodyTemp, timestamp };

        // Dynamic Collection Routing based on deviceID
        const telemetryColl = db.collection(`${deviceID}_telemetry`);
        const configsColl = db.collection(`${deviceID}_device_configs`);
        const aiResponseColl = db.collection(`${deviceID}_AI_response`);

        // Step A: Save incoming data point
        await telemetryColl.insertOne(currentReadings);

        // Step B: Fetch or Initialize Device Threshold Configurations
        let thresholds = await configsColl.findOne({});
        if (!thresholds) {
            thresholds = { ...DEFAULT_CONFIGS, createdAt: timestamp };
            await configsColl.insertOne(thresholds);
        }

        // Step C: Pull historical trend context (last 10 readings)
        const history = await telemetryColl
            .find({})
            .sort({ timestamp: -1 })
            .limit(10)
            .toArray();
        
        // Reverse history to display oldest to newest chronologically for the LLM
        history.reverse(); 

        // Step D: Evaluate against Groq Llama 3 API
        const aiResult = await queryGroqAI(currentReadings, history, thresholds);

        // Step E: Persist AI Assessment
        const aiLog = {
            timestamp,
            telemetryData: currentReadings,
            aiAnalysis: aiResult
        };
        await aiResponseColl.insertOne(aiLog);

        // Step F: Tally thresholds and check if email notification is required
        const crossedThresholds = verifyThresholdBreaches(currentReadings, thresholds);
        
        if (aiResult.status === 'Warning' || aiResult.status === 'Alert' || crossedThresholds.length > 0) {
            await triggerAlertEmail(deviceID, currentReadings, aiResult, crossedThresholds);
        }

        res.status(200).json({
            message: "Telemetry processed cleanly.",
            aiResponse: aiResult
        });

    } catch (error) {
        console.error("System pipeline failure processing data packet:", error);
        res.status(500).json({ error: "Internal Server Pipeline Exception" });
    }
});

// 4. Helper Function: Validate thresholds manually
function verifyThresholdBreaches(current, config) {
    const breaches = [];
    if (current.heartRate < config.heartRate.min || current.heartRate > config.heartRate.max) {
        breaches.push(`Heart Rate (${current.heartRate} BPM) out of stable bounds [${config.heartRate.min}-${config.heartRate.max}]`);
    }
    if (current.spo2 < config.spo2.min || current.spo2 > config.spo2.max) {
        breaches.push(`SpO2 (${current.spo2}%) dropped below or exceeded limits [${config.spo2.min}-${config.spo2.max}]`);
    }
    if (current.bodyTemp < config.bodyTemp.min || current.bodyTemp > config.bodyTemp.max) {
        breaches.push(`Body Temp (${current.bodyTemp}°C) out of stable bounds [${config.bodyTemp.min}-${config.bodyTemp.max}]`);
    }
    return breaches;
}

// 5. Helper Function: Streamlined Groq Call using Structured JSON Mode
async function queryGroqAI(current, history, thresholds) {
    const systemPrompt = `You are a medical diagnostic assistant monitoring remote telemetry systems.
Analyze the user's historical trend (last 10 readings sequentially) alongside their custom physiological boundaries to establish structural risk.

You MUST respond strictly with a valid raw JSON object matching the following structure exactly, with absolutely no surrounding markdown, code fences, or text:
{
  "status": "Normal" | "Warning" | "Alert",
  "remark": "Your critical summary detailing performance trends, systemic spikes, or health warnings."
}

Categorization Criteria:
- "Normal": Stable metrics resting comfortably within standard bounds.
- "Warning": Subtle anomalous escalation trends observed, or metrics hanging closely to structural limits.
- "Alert": Severe threshold violation or immediate critical trend shifts demanding medical review.`;

    const userPrompt = `
Threshold Bounds: ${JSON.stringify(thresholds)}
Historical Vitals Stream (Oldest to Newest): ${JSON.stringify(history)}
Latest Check-in Metrics: ${JSON.stringify(current)}`;

    try {
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ],
                response_format: { type: "json_object" },
                temperature: 0.2
            })
        });

        const rawData = await response.json();
        return JSON.parse(rawData.choices[0].message.content);
    } catch (err) {
        console.error("Failed executing Groq API validation sequence:", err);
        return { status: "Warning", remark: "System failure parsing AI diagnostics. Processing fallback validation rules standardly." };
    }
}

// 6. Helper Function: Nodemailer Engine
async function triggerAlertEmail(deviceID, current, aiResult, breaches) {
    const subject = `[Vitals Alert] status change detected on ${deviceID}: ${aiResult.status}`;
    const body = `
    Device Identification Token: ${deviceID}
    System Status Flag: ${aiResult.status}
    
    Trigger Discovered:
    ${breaches.length > 0 ? breaches.join('\n    ') : 'AI identified high-risk metric variance trends.'}
    
    Current Packet Values:
    - Heart Rate: ${current.heartRate} BPM
    - SpO2 Level: ${current.spo2}%
    - Body Temperature: ${current.bodyTemp}°C
    
    Clinical AI Insight Remark:
    ${aiResult.remark}
    
    Timestamp: ${new Date().toISOString()}
    `;

    try {
        await transporter.sendMail({
            from: process.env.EMAIL_USER,
            to: process.env.NOTIFICATION_RECEIVER,
            subject: subject,
            text: body
        });
        console.log(`Alert dispatch compiled and issued to receiver successfully for device ${deviceID}.`);
    } catch (err) {
        console.error("Critical error firing off automated SMTP mail message:", err);
    }
}

app.listen(PORT, () => {
    console.log(`Telemetry pipeline live on port ${PORT}`);
});
