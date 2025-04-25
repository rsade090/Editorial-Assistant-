import json
from src.utils.utils import *
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError

BASE = "https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices"

class JSPCrawler:

    def __enter__(self):
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(headless=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.browser.close()
        self.pw.stop()

    def scrape(self, url: str, section: str):
        results = []
        page = self.browser.new_page()
        print(f"Loading {section!r} page …")
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1000)

        # exactly same “Expand all sections” click
        try:
            page.click("button:has-text('Expand all sections')", timeout=5000)
            page.wait_for_timeout(1000)
        except TimeoutError:
            print("Could not find expand button. Continuing...")

    
        toggles = page.locator("[aria-expanded]")
        total = toggles.count()
        print(f"Found {total} collapsible blocks in {section!r}.")

        for i in range(total):
            toggle = toggles.nth(i)
            title = toggle.inner_text().strip()
            panel_id = toggle.get_attribute("aria-controls")
            if not panel_id:
                continue

            panel = page.locator(f"#{panel_id}")
            page.wait_for_timeout(200)
            content = panel.inner_text().strip()

            results.append({
                "content_section": section,
                "content_subsection": title,
                "content": content,
                "url": url
            })

        page.close()
        return results

def url_crawler():
    
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate and wait for the JSP headings to appear
        print("\nNavigating to JSP page…")
        page.goto(BASE, wait_until="domcontentloaded", timeout=30000)
        try:
            page.wait_for_selector("h3.policy-title", timeout=20000)
            print("\nJSP section headings detected")
        except TimeoutError:
            print("\nTimeout waiting for headings. Exiting.")
            browser.close()
            return
        
        # Extract all heading texts
        titles = page.eval_on_selector_all(
            "h3.policy-title",
            "els => els.map(el => el.textContent.trim())"
        )
        browser.close()
        titles = list(set(titles))

    print(f"\nFound {len(titles)} section titles:")
    for t in titles:
        print(t)


    SLUG_OVERRIDES = {
    "opinion": "Opinion",
    "language": "Language",
    "war-terror-and-natural-disasters": "war-terror-natural-disasters",
    "user-generated-content-ugc": "user-generated-content",
}
    # Build and verify URLs
    #print("\nVerifying section URLs:")
    
    valid_urls = []
    
    for t in titles:
        slug = slugify(t)
        slug = SLUG_OVERRIDES.get(slug, slug)
        url = f"{BASE}/{slug}"
        try:
            r = requests.head(url, allow_redirects=True, timeout=5)
            if r.status_code == 200:
                valid_urls.append(url)
                status = "200 OK"
            else:
                status = f"{r.status_code}"
        except Exception as e:
            status = f"Error: {e}"
        
    print("\nValid JSP section URLs:\n")
    valid_urls.sort()
    for u in valid_urls:
        print(u)

    return valid_urls

def crawler():

    urls = url_crawler()
    if not urls:
        print("No URLs found. Exiting.")
        return
    all_sections = []
    with JSPCrawler() as JSPcrawler:
        for url in urls:
            slug = url.rstrip("/").split("/")[-1]
            section_name = slug_to_title(slug)
            section_data = JSPcrawler.scrape(url, section_name)
            all_sections.extend(section_data)

    output_path = Path("data/guidelines.json")
    output_path.write_text(
        json.dumps(all_sections, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {len(all_sections)} total sections to {output_path}")

if __name__ == "__main__":
    crawler()
