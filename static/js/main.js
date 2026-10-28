// Live credit updater
async function refreshCredits() {
    try {
        const res = await fetch('/api/check-credits');
        if (!res.ok) return;
        const data = await res.json();
        const el = document.getElementById('navCredits');
        if (el) el.innerText = data.credits;
    } catch (e) {}
}

document.addEventListener('DOMContentLoaded', () => {
    refreshCredits();
    setInterval(refreshCredits, 30000);
});
