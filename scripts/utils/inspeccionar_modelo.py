import json, textwrap

with open('../model_pol.json','r') as f:
    j = json.load(f)

def find_lambdas(obj, path="root"):
    found = []
    if isinstance(obj, dict):
        # Identificar capas Lambda en estructuras tipicas de Keras
        if obj.get("class_name") == "Lambda":
            cfg = obj.get("config", {})
            fun = cfg.get("function", None)
            # Algunas exportaciones guardan la funcion en distintos campos
            # Ej: {"function": ["...codigo..."]} o {"function": {"config": "..."}}
            # Intentamos extraer texto legible
            code_snippets = []
            if isinstance(fun, str):
                code_snippets.append(fun)
            elif isinstance(fun, list):
                for x in fun:
                    if isinstance(x, str) and x.strip():
                        code_snippets.append(x)
            elif isinstance(fun, dict):
                for k,v in fun.items():
                    if isinstance(v, str) and v.strip():
                        code_snippets.append(v)
            found.append({
                "path": path,
                "name": cfg.get("name", "<sin-nombre>"),
                "snippet": "\n---\n".join(textwrap.shorten(s, width=500, placeholder=" ...") for s in code_snippets) or "<no visible text>"
            })
        for k,v in obj.items():
            found.extend(find_lambdas(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            found.extend(find_lambdas(v, f"{path}[{i}]"))
    return found

lambdas = find_lambdas(j)
print(f"Capas Lambda encontradas: {len(lambdas)}")
for i,lam in enumerate(lambdas, 1):
    print("="*80)
    print(f"[{i}] path: {lam['path']}")
    print(f"    name: {lam['name']}")
    print("    funcion (recortada):")
    print(lam["snippet"])

