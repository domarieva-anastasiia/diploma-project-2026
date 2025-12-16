from flask import Blueprint, jsonify

api_bp = Blueprint("api", __name__)

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "API is running"
    })


# Super-resolution (тимчасова версія)
@api_bp.route("/super-resolution", methods=["POST"])
def super_resolution():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"}), 400

    image_file = request.files["image"]

    # створюємо папку results, якщо її немає
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # унікальна назва файлу
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)

    # зберігаємо файл
    image_file.save(filepath)

    return jsonify({
        "success": True,
        "message": "Image received",
        "image_url": f"/results/{filename}"
    })