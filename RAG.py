import requests
import base64
import uuid
import os
import time


# INTEG
APP_CLIENT_ID = "695S322eDfjuWUlEyepeNGzh2esa"
APP_CLIENT_SECRET = "G4vgQZaQlY1JyxLsOqux7MqIzvka"
IDS_AUDIENCE = "695S322eDfjuWUlEyepeNGzh2esa"

API_ENDPOINT = "https://ids-integ.ailab.gehealthcare.com/api/v1"
file_directory = "files"
access_token = ''
documents_list = []


def get_access_token():
    idam_url = "https://idam.gehealthcloud.io/oauth2/token"
    basic_auth_creds = f"{APP_CLIENT_ID}:{APP_CLIENT_SECRET}"
    basic_auth_creds = base64.b64encode(basic_auth_creds.encode()).decode("ascii")
    payload = (
        f"grant_type=client_credentials&scope={uuid.uuid4()}&audience={IDS_AUDIENCE}"
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {basic_auth_creds}",
    }
    response = requests.request(
        "POST", idam_url, headers=headers, data=payload, verify=False
    )
    print(response.json()["access_token"])
    return response.json()["access_token"]


def create_collection(title, description):
    CREATE_COLLECTION_URL = f"{API_ENDPOINT}/collections"
    payload = {
        "title": title,
        "description": description,
        "status": "active",
        "model_id": "model_ada002_beta_embeddings",
    }

    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    response = requests.post(
        CREATE_COLLECTION_URL, headers=headers, json=payload, verify=False
    )

    if response.status_code == 200:
        result = response.json()["response"].split(":")[1].strip()
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def create_batch(collection_id, files_list, batch_name, batch_description):
    CREATE_BATCH_ENDPOINT = f"{API_ENDPOINT}/batch/create"
    print(files_list)
    payload = {
        "batch_name": batch_name,
        "batch_description": batch_description,
        "collection_id": collection_id,
        "file_list": files_list,
        "metadata": {},
    }

    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}"
    }

    try:

        response = requests.post(
            CREATE_BATCH_ENDPOINT, headers=headers, json=payload, verify=False
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: {response.json()}")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")


def upload_file_to_s3(upload_details):
    presignedUrl = upload_details["urls"]
    file_path = upload_details["file_path"]
    with open(file_path, "rb") as file:
        files = {"file": (file_path, file)}
        response = requests.request(
            "POST",
            presignedUrl["url"],
            files=files,
            verify=False,
            data=presignedUrl["fields"],
        )
        if response.status_code == 204:
            print(response)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None


def get_batch_status(batch_id):
    BATCH_DETAILS_ENDPOINT = f"{API_ENDPOINT}/batch/{batch_id}"

    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}"
    }

    response = requests.get(BATCH_DETAILS_ENDPOINT, headers=headers, verify=False)
    print(response.json(), 'aaa')
    success_status_list = []
    failed_status_list = []
    if response.status_code == 200:
        result = response.json()
        total_files = len(result["files"])
        for key, value in result["files"].items():
            if key not in documents_list:
                documents_list.append(key)
            if value["status"] == "success":
                status = {"file_name": value["file_path"], "status": value["status"]}
                success_status_list.append(status)
            if value["status"] == "failed":
                status = {
                    "file_name": value["file_path"],
                    "status": value["status"],
                    "error": value["error"],
                }
                failed_status_list.append(status)

        if len(success_status_list) == total_files:
            return {"message": "All files uploaded successfully"}
        elif len(failed_status_list) + len(success_status_list) == total_files:
            return {
                "message": "Some files failed",
                "failed_file_details": failed_status_list,
            }
        else:
            time.sleep(20)
            return get_batch_status(batch_id)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def query_collection(collection_id, message):
    QUERY_COLLECTION_ENDPOINT = f"{API_ENDPOINT}/chat/conversation"

    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    payload = {
        "collection_id": collection_id,
        "message": message,
    }

    response = requests.post(
        QUERY_COLLECTION_ENDPOINT, headers=headers, json=payload, verify=False
    )

    if response.status_code == 200:
        result = response.json()

        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# DELETE DOCUMENT AND COLLECTION AND VERIFY

def delete_document_soft(collection_id, document_id):
    DELETE_DOCUMENTS_ENDPOINT = f"{API_ENDPOINT}/collections/{collection_id}/documents/{document_id}?soft_delete=true"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    response = requests.request("DELETE", headers=headers, url=DELETE_DOCUMENTS_ENDPOINT, verify=False)
    if response.status_code == 200:
        return response.json()


def delete_document_hard(collection_id, document_id):
    DELETE_DOCUMENTS_ENDPOINT_HARD = f"{API_ENDPOINT}/collections/{collection_id}/documents/{document_id}?soft_delete=false"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    response = requests.request("DELETE", headers=headers, url=DELETE_DOCUMENTS_ENDPOINT_HARD, verify=False)

    if response.status_code == 200:
        return response.json()


def delete_collection_soft(collection_id):
    DELETE_COLLECTION_ENDPOINT = f"{API_ENDPOINT}/collections/{collection_id}?soft_delete=true"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    response = requests.request("DELETE", headers=headers, url=DELETE_COLLECTION_ENDPOINT, verify=False)

    if response.status_code == 200:
        return response.json()


def delete_collection_hard(collection_id):
    DELETE_COLLECTION_ENDPOINT_HARD = f"{API_ENDPOINT}/collections/{collection_id}?soft_delete=false"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    response = requests.request("DELETE", headers=headers, url=DELETE_COLLECTION_ENDPOINT_HARD, verify=False)

    if response.status_code == 200:
        return response.json()


def init_ingestion_and_query():
    collection_name = "Ashok-RAG"
    collection_description = "Ashok-RAG"
    batch_name = "Ashok-RAG-Name"
    batch_description = "Ashok-RAG-Description"
    query_message = "Summarise the document"
    # Create Collection
    collection_id = create_collection(collection_name, collection_description)
    print(collection_id, 'COLLECTION ID')


    # # List of files inside the mentioned directory
    cwd = os.getcwd()
    files_path = os.path.join(cwd, file_directory)
    files = []
    for file in os.listdir(files_path):
        files.append(file)

    # Create Batch based on collection ID created with list of files inside the directory
    batch_details = create_batch(collection_id, files, batch_name, batch_description)
    batch_id = batch_details["batch_id"]
    print(batch_id)
    batch_id = batch_details["batch_id"]
    for value in batch_details["pre_signed_urls"]:
        file_path = os.path.join(files_path, value["file_name"])
        upload_details = {"file_path": file_path, "urls": value}
        upload_file_to_s3(upload_details)

    # Get Batch Status for documents uploaded
    time.sleep(5)
    status = get_batch_status(batch_id)
    print(status, "STATUS")

    time.sleep(5)

    # Query the documents
    response = query_collection(collection_id, query_message)
    print(response)
    return collection_id


def delete_flow(collection_id):
    for document in documents_list:
        response = delete_document_soft(collection_id, document)
        print(response)

    # Here the expectation is it should not answer from that document
    response = query_collection(collection_id, "Data in Transit Security")
    print(response)

    for document in documents_list:
        response = delete_document_hard(collection_id, document)
        print(response)

    # Here the expectation is it should not answer from that document
    response = query_collection(collection_id, "Data in Transit Security")
    print(response)

    response = delete_collection_soft(collection_id)
    print(response)

    response = delete_collection_hard(collection_id)
    print(response)


def main():
    init_ingestion_and_query()


if __name__ == "__main__":
    main()