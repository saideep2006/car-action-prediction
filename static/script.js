const ids = [
  "speed","acceleration","steering_angle","distance_ahead",
  "lane_position","traffic_density","traffic_light","obstacle_detected",
  "road_condition","weather","road_curvature"
];

const $ = id => document.getElementById(id);

function getPayload() {
  const payload = {};
  ids.forEach(id => payload[id] = $(id).value);
  ["speed","acceleration","steering_angle","distance_ahead"].forEach(k => payload[k] = Number(payload[k]));
  return payload;
}

function updateSnapshot() {
  $("snapSpeed").textContent = `${$("speed").value} km/h`;
  $("snapAccel").textContent = `${$("acceleration").value} m/s²`;
  $("snapDistance").textContent = `${$("distance_ahead").value} m`;
  $("snapTraffic").textContent = $("traffic_density").value;
  $("snapLight").textContent = $("traffic_light").value;
  $("snapObstacle").textContent = $("obstacle_detected").value;
  $("snapLane").textContent = $("lane_position").value;
}

ids.forEach(id => $(id).addEventListener("input", updateSnapshot));
ids.forEach(id => $(id).addEventListener("change", updateSnapshot));

const scenarios = {
  red: {speed:50, acceleration:0, steering_angle:0, distance_ahead:8, lane_position:"CENTER", traffic_density:"HIGH", traffic_light:"RED", obstacle_detected:"NO", road_condition:"DRY", weather:"CLEAR", road_curvature:"STRAIGHT"},
  highway: {speed:60, acceleration:0.5, steering_angle:0, distance_ahead:80, lane_position:"CENTER", traffic_density:"LOW", traffic_light:"GREEN", obstacle_detected:"NO", road_condition:"DRY", weather:"CLEAR", road_curvature:"STRAIGHT"},
  curve: {speed:35, acceleration:0, steering_angle:-15, distance_ahead:50, lane_position:"CENTER", traffic_density:"LOW", traffic_light:"NONE", obstacle_detected:"NO", road_condition:"DRY", weather:"CLEAR", road_curvature:"LEFT"}
};

document.querySelectorAll(".scenario").forEach(btn => {
  btn.addEventListener("click", () => {
    const data = scenarios[btn.dataset.scenario];
    ids.forEach(id => $(id).value = data[id]);
    updateSnapshot();
    $("errorBox").hidden = true;
  });
});

function actionClass(action) {
  if (["BRAKE","STOP"].includes(action)) return "red";
  if (action === "ACCELERATE") return "green";
  if (action.includes("LANE")) return "blue";
  return "yellow";
}

function arrowFor(action) {
  return {
    "TURN_LEFT":"←", "TURN_RIGHT":"→",
    "CHANGE_LANE_LEFT":"↖", "CHANGE_LANE_RIGHT":"↗",
    "ACCELERATE":"↑", "BRAKE":"↓", "STOP":"■",
    "MAINTAIN_SPEED":"↑"
  }[action] || "↑";
}

function updateSimulation(action) {
  $("directionArrow").textContent = arrowFor(action);
  $("directionArrow").style.color = actionClass(action) === "red" ? "#ff5c70" : "#35d6ff";
  $("car").querySelectorAll(".tail").forEach(x => x.classList.toggle("on", ["BRAKE","STOP"].includes(action)));
  const light = $("traffic_light").value;
  ["lightRed","lightYellow","lightGreen"].forEach(id => $(id).classList.remove("active"));
  if (light !== "NONE") $("light" + light[0] + light.slice(1).toLowerCase()).classList.add("active");
  $("simText").textContent = `${action.replaceAll("_"," ")} — visual simulation only`;
}

$("predictBtn").addEventListener("click", async () => {
  const btn = $("predictBtn");
  const error = $("errorBox");
  error.hidden = true;
  btn.disabled = true;
  btn.innerHTML = "ANALYZING <span>…</span>";

  try {
    const payload = getPayload();
    if (payload.speed < 0 || payload.speed > 150 || payload.acceleration < -10 || payload.acceleration > 10 ||
        payload.steering_angle < -45 || payload.steering_angle > 45 || payload.distance_ahead < 0 || payload.distance_ahead > 200) {
      throw new Error("Please keep numeric values within the displayed ranges.");
    }

    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");

    const card = $("predictionCard");
    card.className = `prediction-card ${actionClass(data.action)}`;
    $("actionText").textContent = data.action.replaceAll("_"," ");
    $("confidenceText").textContent = data.confidence == null ? "N/A" : `${data.confidence}%`;
    $("messageText").textContent = data.message;
    updateSimulation(data.action);
  } catch (err) {
    error.textContent = err.message || "Something went wrong. Please try again.";
    error.hidden = false;
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'PREDICT ACTION <span>→</span>';
  }
});

async function loadMetrics() {
  try {
    const response = await fetch("/metrics");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);

    const best = data.best_model_metrics;
    $("modelStats").innerHTML = `
      <div><span>Algorithm</span><b>${data.algorithm}</b></div>
      <div><span>Task</span><b>${data.task}</b></div>
      <div><span>Classes</span><b>${data.number_of_classes}</b></div>
      <div><span>Training Samples</span><b>${data.training_samples.toLocaleString()}+</b></div>
      <div><span>Features</span><b>${data.features.length}</b></div>
      <div><span>Accuracy</span><b>${(best.accuracy*100).toFixed(2)}%</b></div>
    `;

    const rows = Object.entries(data.results).map(([name,m]) => `
      <tr><td>${name}</td><td>${(m.accuracy*100).toFixed(2)}%</td><td>${(m.precision*100).toFixed(2)}%</td><td>${(m.recall*100).toFixed(2)}%</td><td>${(m.f1_score*100).toFixed(2)}%</td></tr>
    `).join("");
    $("metricsTable").innerHTML = rows;

    const ctx = $("accuracyChart");
    new Chart(ctx, {
      type:"bar",
      data:{labels:Object.keys(data.results),datasets:[{label:"Accuracy %",data:Object.values(data.results).map(x=>x.accuracy*100)}]},
      options:{responsive:true,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,max:100}}}
    });
  } catch (e) {
    $("modelStats").innerHTML = `<div>Metrics unavailable. Run <b>python train_model.py</b> first.</div>`;
    $("metricsTable").innerHTML = `<tr><td colspan="5">Training metrics are unavailable.</td></tr>`;
  }
}

updateSnapshot();
loadMetrics();
