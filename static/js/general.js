document.addEventListener("DOMContentLoaded", () => {
    initLoadingButtonsForm()
});

function initLoadingButtonsForm() {
    document.querySelectorAll("form").forEach(form => {
        const button = form.querySelector(".loading-button-form");
        if (!button) return;

        form.addEventListener("submit", function (event) {
            // Let native validation pass first
            if (!form.checkValidity()) return;

            event.preventDefault();  // prevent the default fast submission

            // Show spinner
            const originalText = button.innerHTML;
            button.innerHTML = `
                <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                Processing...
            `;
            button.disabled = true;

            // Small delay to allow DOM update before submit
            setTimeout(() => {
                form.submit();  // manually submit the form
            }, 100);  // allow browser to render spinner
        });
    });
}
