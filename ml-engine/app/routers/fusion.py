from fastapi import APIRouter

router = APIRouter()


@router.post("/")
def fusion():
    return {"message": "Fusion module skeleton"}
