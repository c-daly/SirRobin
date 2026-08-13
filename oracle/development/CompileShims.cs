// Compile-only Unity surface for the unreachable CreatureBuilder half of the exact donor BodyGraph.cs.
// The oracle calls only BodyGenome.Measure and SwimEval's Unity-light numerical seams.
namespace UnityEngine
{
    public enum PrimitiveType { Cube }

    public struct Color
    {
        public float r, g, b, a;
        public Color(float r, float g, float b, float a = 1f)
        { this.r = r; this.g = g; this.b = b; this.a = a; }
    }

    public class Object { }

    public class Transform : Object
    {
        public Vector3 localPosition, localScale, position, up;
        public Quaternion localRotation;
        public void SetParent(Transform parent, bool worldPositionStays) { }
        public void SetPositionAndRotation(Vector3 p, Quaternion q) { position = p; }
        public Vector3 InverseTransformPoint(Vector3 p) => p;
        public Vector3 InverseTransformDirection(Vector3 v) => v;
    }

    public class GameObject : Object
    {
        public readonly Transform transform = new();
        public GameObject(string name = "") { }
        public static GameObject CreatePrimitive(PrimitiveType type) => new();
        public T AddComponent<T>() where T : new() => new();
        public T GetComponent<T>() where T : new() => new();
    }

    public class Rigidbody : Object
    {
        public bool useGravity;
        public float angularDamping, mass;
    }

    public class Renderer : Object { public Material? sharedMaterial; }
    public class Material : Object
    {
        public Color color;
        public Material(Shader shader) { }
    }
    public class Shader : Object { public static Shader Find(string name) => new(); }
}

namespace ProceduralWorld.Life
{
    public struct AeroSurface
    {
        public UnityEngine.Vector3 localPos, localNormal;
        public float area;
    }
}
