// --- Helpers ---
function qi(id){ return document.getElementById(id); }
function navTo(href){ window.location.href = href; }

// --- THE NEW "BRIDGE" FUNCTIONS ---

// This function is called for non-audio results (tapping, memorizing, etc.)
// IT NOW SENDS DATA TO THE SERVER.
function setResult(taskIdStr, payload) {
  console.log(`Sending non-audio result for ${taskIdStr} to server...`);
  const formData = new FormData();
  formData.append('task_id_str', taskIdStr);
  formData.append('payload', JSON.stringify(payload));
  
  fetch('/api/save_task_result', {
    method: 'POST',
    body: formData
  }).catch(console.error);
}

// This function is called when you save a recording.
// IT NOW UPLOADS THE AUDIO TO THE SERVER INSTEAD OF DOWNLOADING IT.
function downloadBlob(blob, filename) {
  console.log(`Uploading ${filename} to server...`);
  const taskIdStr = filename.split('.')[0]; // e.g., "task1_picture"
  
  const formData = new FormData();
  formData.append('task_id_str', taskIdStr);
  formData.append('audio_blob', blob, filename);

  fetch('/api/process_task_audio', {
    method: 'POST',
    body: formData
  }).catch(console.error);
}

// --- MicRecorder Class (from your files) ---
class MicRecorder {
  constructor(progressEl, timerEl) {
    this.stream = null; this.rec = null; this.chunks = [];
    this.startTime = null; this.timerInterval = null;
    this.progressEl = progressEl; this.timerEl = timerEl;
  }

  async start() {
    if (!navigator.mediaDevices) { alert("Microphone not supported."); return; }
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.ondataavailable = e => { if (e.data && e.data.size) this.chunks.push(e.data); };
    this.rec.start();
    this.startTime = Date.now();
    if (this.timerEl) this.timerEl.textContent = "0s";
    if (this.progressEl) this.progressEl.value = 0;
    this.timerInterval = setInterval(() => {
      const secs = Math.floor((Date.now() - this.startTime) / 1000);
      if (this.timerEl) this.timerEl.textContent = secs + "s";
      if (this.progressEl) this.progressEl.value = secs;
    }, 1000);
    return true;
  }

  async stop() {
    if (!this.rec) return null;
    return new Promise(res => {
        this.rec.onstop = () => {
            this.stream.getTracks().forEach(t => t.stop());
            clearInterval(this.timerInterval);
            const blob = new Blob(this.chunks, { type: "audio/webm" });
            const url = URL.createObjectURL(blob);
            res({ blob, url, durSec: Math.floor((Date.now() - this.startTime) / 1000) });
        };
        this.rec.stop();
    });
  }
}

// Auto mic prompt
window.addEventListener("DOMContentLoaded", async () => {
  if(document.querySelector("[data-automic='true']")){
    try {
      const s = await navigator.mediaDevices.getUserMedia({audio:true});
      s.getTracks().forEach(t=>t.stop());
    } catch(err) { console.warn("Mic permission not granted yet."); }
  }
});