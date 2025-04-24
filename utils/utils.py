import re 

def slug_to_title(slug: str) -> str:
    return re.sub(r"-+", " ", slug).title()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)   
    text = re.sub(r"\s+", "-", text)       
    return text.strip("-")