/**
 * auth-guard.js — SkyGate Instant Auth Guard
 *
 * Load this script on EVERY page EXCEPT auth.html.
 * It must be the FIRST script tag so it runs before the DOM paints,
 * preventing any authenticated content from flashing to unauthenticated users.
 *
 * Usage in protected pages:
 *   <script src="auth-guard.js"></script>   ← first
 *   <script src="shared.js"></script>        ← second
 */
(function() {
    const token = localStorage.getItem('sg_token');
    const path = window.location.pathname;
    const isAuthPage = path.includes('auth.html');

    if (!token && !isAuthPage) {
        // Force login if no token found
        window.location.href = 'auth.html';
    } else if (token && isAuthPage) {
        // Already logged in? Go to dashboard
        window.location.href = 'index.html';
    } else {
        // Everything is fine, let CSS show the body
        document.documentElement.classList.add('authenticated');
    }
})();