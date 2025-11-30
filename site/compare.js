async function loadJSON(path) {
  const res = await fetch(path);
  return await res.json();
}

async function main() {
  const forecast = await loadJSON("../data/forecast.json");
  const real = await loadJSON("../data/real_weather.json");

  const times = forecast.time;

  const tempF = forecast.temp;
  const tempR = real.temp;

  const rainF = forecast.rain;
  const rainR = real.rain;

  const table = document.getElementById("result-table");

  let tempErrors = [];
  let rainErrors = [];

  for (let i = 0; i < times.length; i++) {
    const te = Math.abs(tempF[i] - tempR[i]);
    const re = Math.abs(rainF[i] - rainR[i]);

    tempErrors.push(te);
    rainErrors.push(re);

    const row = `
      <tr>
        <td>${times[i].slice(11,16)}</td>
        <td>${tempF[i]}°C</td>
        <td>${tempR[i]}°C</td>
        <td>${te.toFixed(1)}°C</td>
        <td>${rainF[i]}%</td>
        <td>${rainR[i]}%</td>
        <td>${re.toFixed(1)}%</td>
      </tr>
    `;
    table.insertAdjacentHTML("beforeend", row);
  }

  // 精度スコア (MAPEっぽい)
  const avgTempErr = tempErrors.reduce((a,b)=>a+b)/tempErrors.length;
  const avgRainErr = rainErrors.reduce((a,b)=>a+b)/rainErrors.length;

  document.getElementById("score").innerHTML = `
    今日の精度スコア  
    気温誤差：<b>${avgTempErr.toFixed(2)}℃</b>  
    降水誤差：<b>${avgRainErr.toFixed(2)}%</b>
  `;

  // グラフ
  new Chart(document.getElementById("chart"), {
    type: "line",
    data: {
      labels: times.map(t => t.slice(11,16)),
      datasets: [
        { label: "予測 気温", data: tempF, borderColor: "#ff6b6b" },
        { label: "実測 気温", data: tempR, borderColor: "#4d79ff" },
        { label: "予測 降水%", data: rainF, borderColor: "#ffa94d", yAxisID: 'y1' },
        { label: "実測 降水%", data: rainR, borderColor: "#4dd2c6", yAxisID: 'y1' },
      ],
    },
    options: {
      scales: {
        y: { type: 'linear', position: 'left' },
        y1: { type: 'linear', position: 'right' }
      }
    }
  });
}

main();
