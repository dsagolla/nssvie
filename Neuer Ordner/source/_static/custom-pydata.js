// Change the link of logo to index.html
document.addEventListener("DOMContentLoaded", function() {

    // Find an "<a>" element whose href ends with "contents.html"
    const match = document.querySelector("a[href*='contents.html']");

    // Replace "contents.html" with "index.html"
    if (match){
        match.href = match.href.replace('contents.html', 'index.html')
    }
});
