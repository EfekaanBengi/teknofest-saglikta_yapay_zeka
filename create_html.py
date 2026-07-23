import markdown
import codecs
import re

with codecs.open('AKADEMIK_PIPELINE_RAPORU.md', mode="r", encoding="utf-8") as input_file:
    text = input_file.read()

# Fix image paths for local HTML viewing
text = text.replace('outputs/', './outputs/')

# Convert markdown to html with tables extension
html = markdown.markdown(text, extensions=['tables', 'fenced_code'])

css = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; padding: 40px; max-width: 1000px; margin: auto; color: #333; background-color: #fcfcfc;}
    h1, h2, h3, h4 { color: #2c3e50; margin-top: 30px; }
    h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    table { border-collapse: collapse; width: 100%; margin-top: 20px; margin-bottom: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); background-color: #fff;}
    th, td { border: 1px solid #e0e0e0; padding: 12px; text-align: left; }
    th { background-color: #f8f9fa; font-weight: bold; color: #2c3e50; }
    tr:nth-child(even) { background-color: #fbfcfc; }
    img { max-width: 100%; height: auto; display: block; margin: 25px auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    code { background-color: #f1f2f6; padding: 3px 6px; border-radius: 4px; font-family: Consolas, monospace; color: #e74c3c;}
    blockquote { border-left: 5px solid #3498db; margin: 1.5em 0; padding: 0.5em 20px; font-style: italic; background: #ebf5fb;}
    hr { border: 0; height: 1px; background: #e0e0e0; margin: 40px 0; }
</style>
"""

js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    // Tüm paragrafları gez ve sadece 4 görsel içerenleri bul
    document.querySelectorAll('p').forEach(p => {
        let imgs = p.querySelectorAll('img');
        // Eğer paragraf içinde 4 görsel varsa ve metin yoksa
        if (imgs.length === 4 && p.textContent.trim() === '') {
            // SHAP Force plotları yatayda çok geniş olduğu için onları grid yapmaktan kaçın
            let isForcePlot = Array.from(imgs).some(img => img.src.includes('force_fn'));
            if (!isForcePlot) {
                p.style.display = 'grid';
                p.style.gridTemplateColumns = '1fr 1fr'; // 2 kolon
                p.style.gap = '20px';
                imgs.forEach(img => { 
                    img.style.margin = '0'; 
                    img.style.width = '100%'; 
                    img.style.height = 'auto';
                    img.style.objectFit = 'contain';
                });
            }
        }
    });
});
</script>
"""

final_html = f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n{css}\n</head>\n<body>\n{html}\n{js}\n</body>\n</html>"

with codecs.open('AKADEMIK_PIPELINE_RAPORU.html', mode="w", encoding="utf-8") as output_file:
    output_file.write(final_html)

print("HTML created successfully with 2x2 grid formatting!")
