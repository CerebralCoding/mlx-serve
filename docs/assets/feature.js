// feature.js — shared behavior for mlx-serve deep-dive pages.

// Scroll reveal
(function () {
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
  document.querySelectorAll('.reveal').forEach(function (el) { observer.observe(el); });
})();

// Copy-to-clipboard for code blocks
function copyCode(btn) {
  var pre = btn.closest('.code-block').querySelector('pre');
  navigator.clipboard.writeText(pre.textContent).then(function () {
    var old = btn.innerHTML;
    btn.innerHTML = '<svg viewBox="0 0 24 24" style="stroke:#30d158"><polyline points="20 6 9 17 4 12"/></svg>';
    setTimeout(function () { btn.innerHTML = old; }, 1400);
  });
}

// Screenshots that haven't been captured yet fall back to a designed
// "coming soon" tile instead of a broken-image icon.
(function () {
  function markMissing(img) {
    var shot = img.closest('.shot');
    if (shot) shot.classList.add('missing');
  }
  document.querySelectorAll('.shot img').forEach(function (img) {
    img.addEventListener('error', function () { markMissing(img); });
    if (img.complete && img.naturalWidth === 0) markMissing(img);
  });
})();
