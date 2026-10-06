// Public reading controls; editing and local annotation state are not published.
function toggleTheme() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
}
function openLightbox(src) {
  const image = document.getElementById('lightbox-img');
  const box = document.getElementById('img-lightbox');
  if (image && box) { image.src = src; box.style.display = 'flex'; }
}
function closeLightbox() {
  const box = document.getElementById('img-lightbox');
  if (box) box.style.display = 'none';
}
async function copyCode(button) {
  const code = button.dataset.code?.replace(/\\n/g, '\n') || button.closest('.code-window')?.querySelector('pre code')?.textContent;
  if (!code) return;
  try {
    await navigator.clipboard.writeText(code);
    const label = button.textContent;
    button.textContent = '已复制';
    setTimeout(() => { button.textContent = label; }, 1500);
  } catch { button.textContent = '请选中代码复制'; }
}
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeLightbox(); });
window.addEventListener('scroll', () => {
  const bar = document.getElementById('reading-progress');
  const height = document.documentElement.scrollHeight - innerHeight;
  if (bar) bar.style.width = `${height > 0 ? scrollY / height * 100 : 0}%`;
}, { passive: true });
