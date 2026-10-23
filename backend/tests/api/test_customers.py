from tests.conftest import customer_payload


def test_health_and_root(client):
    assert client.get("/health").json() == {"status": "healthy", "environment": "test"}
    assert client.get("/").json()["docs"] == "/docs"


def test_create_and_get_customer(client):
    res = client.post("/api/v1/customers", json=customer_payload())
    assert res.status_code == 201
    body = res.json()
    assert body["customer_id"] == "C-0001" and body["id"] > 0

    got = client.get("/api/v1/customers/C-0001")
    assert got.status_code == 200
    assert got.json()["monthly_charges"] == 85.7


def test_duplicate_customer_is_rejected(client):
    client.post("/api/v1/customers", json=customer_payload())
    assert client.post("/api/v1/customers", json=customer_payload()).status_code == 409


def test_unknown_customer_returns_404(client):
    assert client.get("/api/v1/customers/NOPE").status_code == 404
    assert client.put("/api/v1/customers/9999", json={"tenure": 3}).status_code == 404


def test_invalid_payloads_return_422(client):
    assert client.post("/api/v1/customers", json=customer_payload(tenure=99)).status_code == 422
    bad = customer_payload()
    del bad["contract"]
    assert client.post("/api/v1/customers", json=bad).status_code == 422


def test_update_only_changes_sent_fields(client):
    created = client.post("/api/v1/customers", json=customer_payload()).json()
    res = client.put(f"/api/v1/customers/{created['id']}", json={"tenure": 30, "contract": "One year"})
    assert res.status_code == 200
    body = res.json()
    assert (body["tenure"], body["contract"], body["internet_service"]) == (30, "One year", "Fiber optic")


def test_list_is_paginated(client):
    for i in range(5):
        client.post("/api/v1/customers", json=customer_payload(customer_id=f"C-{i}"))
    page = client.get("/api/v1/customers", params={"page": 2, "page_size": 2}).json()
    assert page["total"] == 5
    assert [c["customer_id"] for c in page["customers"]] == ["C-2", "C-3"]


def test_bulk_upserts_existing_customers(client):
    client.post("/api/v1/customers", json=customer_payload(customer_id="C-1", tenure=1))
    res = client.post("/api/v1/customers/bulk", json=[
        customer_payload(customer_id="C-1", tenure=40),
        customer_payload(customer_id="C-2"),
    ])
    assert res.status_code == 201
    assert client.get("/api/v1/customers").json()["total"] == 2
    assert client.get("/api/v1/customers/C-1").json()["tenure"] == 40
