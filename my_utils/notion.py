import requests

def get_notion_dicts(NOTION_API_KEY, DATABASE_ID, headers):
    pages = get_all_database_pages(NOTION_API_KEY, DATABASE_ID, headers)
    
    pages_dict = {}
    for page in pages:
        pages_dict[page["id"]] = page
        
    model_dict = {}
    for page in pages:
        page_id = page["id"]
        property = page["properties"]
        try:
            model_name = property["Model"]['select']['name']
        except: continue
        if model_name not in model_dict:
            model_dict[model_name] = []
        model_dict[model_name].append(page_id)
    return pages_dict, model_dict

def get_all_database_pages(NOTION_API_KEY, DATABASE_ID, headers):    
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    all_pages = []
    has_more = True
    start_cursor = None
    
    while has_more:
        if start_cursor:
            data = {"start_cursor": start_cursor}
        else:
            data = {}
        
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        
        all_pages.extend(response_data["results"])
        has_more = response_data.get("has_more", False)
        start_cursor = response_data.get("next_cursor", None)
    print(f"Total pages: {len(all_pages)}")
    return all_pages

def update_mpjpe(page_id, headers, mpjpe, p_mpjpe):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            "MPJPE": {
                "number": mpjpe
            },
            "P-MPJPE": {
                "number": p_mpjpe
            }
        },
    }

    response = requests.patch(url, headers=headers, json=data)
    return response.json()

def create_notion_page(database_id, headers, properties, children=None):
    url = "https://api.notion.com/v1/pages"
    data = {
        "parent": {"database_id": database_id},
        "properties": properties
    }
    if children:
        data["children"] = children
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def update_notion_page(page_id, headers, properties, children=None):
    print(f"Updating page {page_id}")
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": properties
    }
    if children:
        data["children"] = children
    
    response = requests.patch(url, headers=headers, json=data)
    return response.json()