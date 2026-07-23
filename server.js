require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');
const nodemailer = require('nodemailer');

const app = express();

// Enable CORS for all origins (or restrict to your Vercel URL in production)
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;
let db;

// 1. Initialize MongoDB Connection
const client = new MongoClient(process.env.MONGODB_URI);
async function connectDB() {
    try {
        await client.connect();
        db = client.db('Vitals_Monitoring');
        console.log("Connected successfully to MongoDB");
    } catch (err) {
        console.error("MongoDB connection failed:", err);
        process.exit(1);
    }
}
connectDB();

// 2. Setup Email Transporter
const transporter = nodemailer.createTransport({
    host: 'smtp.gmail.com',
    port: 587,
    secure: false, // true for 465, false for 587 (STARTTLS)
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS // App Password
    },
    tls: {
        rejectUnauthorized: false
    },
    connectionTimeout: 5000,
    greetingTimeout: 5000,
    socketTimeout: 5000
});

// Default configurations
const DEFAULT_CONFIGS = {
    heartRate: { min: 60, max: 100 },
    spo2: { min: 95, max: 100 },
    bodyTemp: { min: 36.0, max: 37.5 }
};

// Helper: Sanitize Mongo ObjectId for JSON responses
function sanitizeDoc(doc) {
    if (!doc) return null;
    const { _id, ...cleanDoc } = doc;
    return { ...cleanDoc, id: _id ? _id.toString() : undefined };
}

// 3. Telemetry Processing Endpoint (Async / Non-Blocking for ESP32)
app.post('/api/telemetry', async (req, res) => {
    const { deviceID, heartRate, spo2, bodyTemp } = req.body;

    if (!deviceID || heartRate === undefined || spo2 === undefined || bodyTemp === undefined) {
        return res.status(400).json({ error: "Missing required fields within telemetry payload." });
    }

    try {
        const timestamp = new Date();
        const currentReadings = { heartRate, spo2, bodyTemp, timestamp };

        // Dynamic Collections
        const telemetryColl = db.collection(`${deviceID}_telemetry`);
        const configsColl = db.collection(`${deviceID}_device_configs`);
        const aiResponseColl = db.collection(`${deviceID}_AI_response`);

        // Step A: Immediately persist incoming telemetry packet
        await telemetryColl.insertOne(currentReadings);

        // Step B: Send HTTP 200 Success ACK to ESP32 IMMEDIATELY
        res.status(200).json({
            status: "success",
            message: "Telemetry saved successfully."
        });

        // Step C: Execute AI processing & Notifications in BACKGROUND
        setImmediate(async () => {
            try {
                // Fetch or initialize configs
                let thresholds = await configsColl.findOne({});
                if (!thresholds) {
                    thresholds = { ...DEFAULT_CONFIGS, createdAt: timestamp };
                    await configsColl.insertOne(thresholds);
                }

                // Get last 10 readings for context
                const history = await telemetryColl
                    .find({})
                    .sort({ timestamp: -1 })
                    .limit(10)
                    .toArray();
                
                history.reverse();

                // Query Groq AI asynchronously
                const aiResult = await queryGroqAI(currentReadings, history, thresholds);

                // Save AI log
                await aiResponseColl.insertOne({
                    timestamp,
                    telemetryData: currentReadings,
                    aiAnalysis: aiResult
                });

                // Check thresholds & send notification if necessary
                const crossedThresholds = verifyThresholdBreaches(currentReadings, thresholds);
                if (aiResult.status === 'Warning' || aiResult.status === 'Alert' || crossedThresholds.length > 0) {
                    await triggerAlertEmail(deviceID, currentReadings, aiResult, crossedThresholds);
                }
            } catch (bgError) {
                console.error("Background AI processing error:", bgError);
            }
        });

    } catch (error) {
        console.error("System pipeline failure processing data packet:", error);
        if (!res.headersSent) {
            res.status(500).json({ error: "Internal Server Pipeline Exception" });
        }
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

// 5. Helper Function: Streamlined Groq Call
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

// --- FRONTEND REST ENDPOINTS ---

// 1. Dynamic List of active devices in MongoDB
app.get('/api/devices', async (req, res) => {
    try {
        const collections = await db.listCollections().toArray();
        const deviceSet = new Set();

        collections.forEach(col => {
            if (col.name.endsWith('_telemetry')) {
                const deviceID = col.name.replace('_telemetry', '');
                deviceSet.add(deviceID);
            }
        });

        res.status(200).json({ devices: Array.from(deviceSet) });
    } catch (error) {
        console.error("Error fetching device list:", error);
        res.status(500).json({ error: "Failed to list active devices." });
    }
});

// 2. Get Live Telemetry, History, AI Remark, and Thresholds for a specific Device
app.get('/api/device/:deviceID', async (req, res) => {
    const { deviceID } = req.params;

    try {
        const telemetryColl = db.collection(`${deviceID}_telemetry`);
        const configsColl = db.collection(`${deviceID}_device_configs`);
        const aiResponseColl = db.collection(`${deviceID}_AI_response`);

        // Fetch latest telemetry
        const latestTelemetry = await telemetryColl.find({}).sort({ timestamp: -1 }).limit(1).toArray();
        
        // Fetch last 20 telemetry readings for charts
        const history = await telemetryColl.find({}).sort({ timestamp: -1 }).limit(20).toArray();
        history.reverse();

        // Fetch latest AI response log
        const latestAI = await aiResponseColl.find({}).sort({ timestamp: -1 }).limit(1).toArray();

        // Fetch device thresholds
        let thresholds = await configsColl.findOne({});
        if (!thresholds) {
            thresholds = DEFAULT_CONFIGS;
        }

        res.status(200).json({
            latest: sanitizeDoc(latestTelemetry[0]),
            history: history.map(sanitizeDoc),
            aiResponse: latestAI[0] ? latestAI[0].aiAnalysis : null,
            thresholds: sanitizeDoc(thresholds)
        });
    } catch (error) {
        console.error(`Error fetching data for ${deviceID}:`, error);
        res.status(500).json({ error: "Failed to retrieve device telemetry data." });
    }
});

// 3. Update Threshold Configurations for a specific Device
app.post('/api/device/:deviceID/config', async (req, res) => {
    const { deviceID } = req.params;
    const { heartRate, spo2, bodyTemp } = req.body;

    if (!heartRate || !spo2 || !bodyTemp) {
        return res.status(400).json({ error: "Invalid threshold payload provided." });
    }

    try {
        const configsColl = db.collection(`${deviceID}_device_configs`);

        const updatedConfig = {
            heartRate: { min: Number(heartRate.min), max: Number(heartRate.max) },
            spo2: { min: Number(spo2.min), max: Number(spo2.max) },
            bodyTemp: { min: Number(bodyTemp.min), max: Number(bodyTemp.max) },
            updatedAt: new Date()
        };

        // Upsert configuration
        await configsColl.replaceOne({}, updatedConfig, { upsert: true });

        res.status(200).json({ message: "Threshold configurations updated successfully!", config: updatedConfig });
    } catch (error) {
        console.error(`Error updating configs for ${deviceID}:`, error);
        res.status(500).json({ error: "Failed to update device configurations." });
    }
});

app.listen(PORT, () => {
    console.log(`Telemetry pipeline live on port ${PORT}`);
});
