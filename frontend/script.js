// 智能观鸟系统 v4 - 前端逻辑（从 index.html 抽离）
let currentPhotos = [];
let currentPhotoIndex = 0;
let currentDate = null;
let lastPhotoSig = "";  // 用于增量刷新：仅当照片集合变化时才重渲染
let currentBackendName = "cpu";  // 当前检测后端（由 /status 同步，作为推荐参数依据）
const BACKEND_INFO = {
    cpu: "推理后端：CPU（通用 · 兼容最好 · 约 0.1 FPS）",
    npu: "推理后端：NPU（RK3566 · 约 1.7 FPS · 约 12× 加速）"
};

// 更新状态和调试信息
async function updateStatus() {
    try {
        const res = await fetch("/status");
        const data = await res.json();

        const badge = document.getElementById("status-badge");
        const text = document.getElementById("status-text");

        text.textContent = data.state;
        badge.className = "status-badge";

        if (data.state === "RUNNING") {
            badge.classList.add("status-running");
        } else if (data.state === "SLEEPING") {
            badge.classList.add("status-sleeping");
        } else {
            badge.classList.add("status-offline");
        }

        document.getElementById("box-count").textContent = data.latest_boxes || 0;
        document.getElementById("save-count").textContent = data.save_counter || 0;
        document.getElementById("frame-count").textContent = data.frame_counter || 0;

        // 后端状态同步（按钮心跳 + 状态文字，不再用“当前后端”文本徽标）
        const backend = (data.backend || "cpu").toLowerCase();
        currentBackendName = backend;
        document.getElementById("btn-cpu").classList.toggle("active-cpu", backend === "cpu");
        document.getElementById("btn-npu").classList.toggle("active-npu", backend === "npu");
        const st = document.getElementById("backend-status-text");
        if (st) st.textContent = BACKEND_INFO[backend] || ("推理后端：" + backend.toUpperCase());
        renderRecommended();
    } catch (e) {
        console.error("状态更新失败:", e);
    }
}

function updateConfDisplay(value) {
    document.getElementById("conf-display").textContent = value;
}

function updateIntervalDisplay(value) {
    document.getElementById("interval-display").textContent = value;
    document.getElementById("interval-badge").textContent = value + "帧";
}

function updateConsecutiveDisplay(value) {
    document.getElementById("consecutive-display").textContent = value;
    document.getElementById("consecutive-badge").textContent = value + "帧";
}

async function updateCamera() {
    const ip = document.getElementById("camera-ip").value;
    try {
        const res = await fetch("/update_camera", {
            method: "POST",
            body: new URLSearchParams({ ip })
        });
        const data = await res.json();
        showToast("摄像头已切换到: " + ip, "success");
    } catch (e) {
        showToast("切换失败", "error");
    }
}

async function updateYOLO() {
    const conf = parseFloat(document.getElementById("yolo-conf").value);
    try {
        const res = await fetch("/update_yolo", {
            method: "POST",
            body: new URLSearchParams({ conf })
        });
        const data = await res.json();
        showToast(data.message, "success");
    } catch (e) {
        showToast("更新失败", "error");
    }
}

async function updateDetectionParams() {
    const interval = parseInt(document.getElementById("detect-interval").value);
    const consecutive = parseInt(document.getElementById("consecutive-frames").value);
    try {
        const res = await fetch("/update_detection_params", {
            method: "POST",
            body: new URLSearchParams({ interval, consecutive })
        });
        const data = await res.json();
        showToast(data.message, "success");
    } catch (e) {
        showToast("更新失败", "error");
    }
}

async function updateBackend(b) {
    try {
        const res = await fetch("/update_backend", {
            method: "POST",
            body: new URLSearchParams({ backend: b })
        });
        const data = await res.json();
        if (data.ok) {
            const label = (data.backend || b).toUpperCase();
            showToast(data.message || `检测后端已切换为 ${label}`, data.fallback ? "warning" : "success");
            await updateStatus();
            await loadRecommended();
        } else {
            showToast(data.message || "切换失败", "error");
        }
    } catch (e) {
        showToast("切换失败", "error");
    }
}

let recommended = null;

async function loadRecommended() {
    try {
        const res = await fetch("/recommended_params");
        recommended = await res.json();
        renderRecommended();
    } catch (e) {
        console.error("加载推荐参数失败:", e);
        const note = document.getElementById("rec-note");
        if (note) note.textContent = "推荐参数加载失败，3 秒后重试…";
        setTimeout(loadRecommended, 3000);  // 失败后自动重试，避免一直卡在“加载中”
    }
}

function currentBackend() {
    return currentBackendName;
}

function renderRecommended() {
    if (!recommended) return;
    const rec = recommended[currentBackend()] || recommended["cpu"];
    document.getElementById("rec-conf").textContent = rec.conf.toFixed(2);
    document.getElementById("rec-interval").textContent = rec.interval + "帧";
    document.getElementById("rec-consecutive").textContent = rec.consecutive + "帧";
    document.getElementById("rec-note").textContent = rec.note;
}

async function applyRecommended() {
    if (!recommended) { showToast("推荐参数未加载", "error"); return; }
    const rec = recommended[currentBackend()] || recommended["cpu"];
    document.getElementById("yolo-conf").value = rec.conf;
    document.getElementById("detect-interval").value = rec.interval;
    document.getElementById("consecutive-frames").value = rec.consecutive;
    updateConfDisplay(rec.conf);
    updateIntervalDisplay(rec.interval);
    updateConsecutiveDisplay(rec.consecutive);
    try {
        await updateYOLO();
        await updateDetectionParams();
        showToast(`已应用 ${currentBackend().toUpperCase()} 推荐参数`, "success");
    } catch (e) {
        showToast("应用失败", "error");
    }
}

async function updateWorktime() {
    const start = parseInt(document.getElementById("start-hour").value);
    const end = parseInt(document.getElementById("end-hour").value);
    try {
        const res = await fetch("/update_worktime", {
            method: "POST",
            body: new URLSearchParams({ start, end })
        });
        const data = await res.json();
        showToast(data.message, "success");
    } catch (e) {
        showToast("更新失败", "error");
    }
}

async function snapshot() {
    try {
        const res = await fetch("/snapshot_simple", { method: "POST" });
        const data = await res.json();
        if (data.ok) {
            showToast("截图已保存", "success");
            lastPhotoSig = "";  // 强制下次刷新立即生效
            refreshPhotos(currentDate);
        } else {
            showToast("截图失败", "error");
        }
    } catch (e) {
        showToast("截图失败", "error");
    }
}

async function loadDates() {
    try {
        const res = await fetch("/api/dates");
        const data = await res.json();

        const selector = document.getElementById("date-selector");
        selector.innerHTML = "";

        if (data.dates.length === 0) {
            selector.innerHTML = "<span style='color: var(--text-secondary); font-size: 0.875rem;'>暂无记录</span>";
            return;
        }

        data.dates.forEach((dateInfo, index) => {
            const chip = document.createElement("div");
            chip.className = "date-chip" + (index === 0 ? " active" : "");
            chip.textContent = dateInfo.display;
            chip.onclick = () => selectDate(dateInfo.date, chip);
            selector.appendChild(chip);
        });

        if (data.dates.length > 0 && !currentDate) {
            currentDate = data.dates[0].date;
            loadPhotos(currentDate);
        }
    } catch (e) {
        console.error("加载日期失败:", e);
    }
}

function selectDate(date, element) {
    currentDate = date;
    document.querySelectorAll(".date-chip").forEach(chip => chip.classList.remove("active"));
    element.classList.add("active");
    loadPhotos(date);
}

// 完整加载（切换日期时调用）
async function loadPhotos(date) {
    if (!date) date = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    currentDate = date;
    lastPhotoSig = "";  // 强制刷新
    await refreshPhotos(date);
}

// 增量刷新：仅在照片集合变化且未打开大图时重渲染，避免闪烁与滚动丢失
async function refreshPhotos(date) {
    if (!date) return;
    if (document.getElementById("image-modal").classList.contains("active")) return;
    try {
        const res = await fetch(`/api/photos?date=${date}&limit=100`);
        const data = await res.json();
        const sig = data.photos.map(p => p.filename).join("|");
        if (sig === lastPhotoSig) return;
        lastPhotoSig = sig;
        renderPhotos(data.photos);
    } catch (e) {
        console.error("刷新照片失败:", e);
    }
}

function renderPhotos(photos) {
    const wall = document.getElementById("photo-wall");
    currentPhotos = photos || [];

    if (currentPhotos.length === 0) {
        wall.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🐦</div>
                <p>该日期暂无照片</p>
            </div>`;
        return;
    }

    wall.innerHTML = "";
    currentPhotos.forEach((photo, index) => {
        const card = document.createElement("div");
        card.className = "photo-card";
        card.onclick = () => openModal(index);

        const timeStr = photo.time.slice(0, 2) + ":" + photo.time.slice(2, 4) + ":" + photo.time.slice(4, 6);
        const sizeKB = (photo.size / 1024).toFixed(1);

        card.innerHTML = `
            <img class="photo-thumb" src="${photo.thumb_url}" loading="lazy" alt="鸟类照片">
            <div class="photo-info">
                <div class="photo-time">${timeStr}</div>
                <div class="photo-meta">${sizeKB} KB</div>
            </div>`;
        wall.appendChild(card);
    });
}

function openModal(index) {
    currentPhotoIndex = index;
    const photo = currentPhotos[index];
    if (!photo) return;
    document.getElementById("modal-image").src = photo.full_url;
    document.getElementById("image-modal").classList.add("active");
}

function closeModal() {
    document.getElementById("image-modal").classList.remove("active");
}

function navigateImage(direction) {
    if (currentPhotos.length === 0) return;
    currentPhotoIndex += direction;
    if (currentPhotoIndex < 0) currentPhotoIndex = currentPhotos.length - 1;
    if (currentPhotoIndex >= currentPhotos.length) currentPhotoIndex = 0;

    const photo = currentPhotos[currentPhotoIndex];
    document.getElementById("modal-image").src = photo.full_url;
}

function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        padding: 1rem 1.5rem;
        background: ${type === "success" ? "var(--accent)" : type === "error" ? "var(--danger)" : "var(--warning)"};
        color: white;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        z-index: 2000;
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = "slideOut 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

document.addEventListener("keydown", (e) => {
    if (!document.getElementById("image-modal").classList.contains("active")) return;
    if (e.key === "Escape") closeModal();
    if (e.key === "ArrowLeft") navigateImage(-1);
    if (e.key === "ArrowRight") navigateImage(1);
});

document.getElementById("image-modal").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeModal();
});

// 动画样式
const style = document.createElement("style");
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// 初始化
setInterval(updateStatus, 1000);
setInterval(() => refreshPhotos(currentDate), 5000);
setInterval(loadDates, 30000);

updateStatus();
loadDates();
loadRecommended();
