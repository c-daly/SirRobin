using System;
using NQ = System.Numerics.Quaternion;
using NV = System.Numerics.Vector3;

namespace UnityEngine
{
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 zero => new(0, 0, 0);
        public static Vector3 right => new(1, 0, 0);
        public static Vector3 up => new(0, 1, 0);
        public static Vector3 forward => new(0, 0, 1);
        public static Vector3 back => new(0, 0, -1);
        public float sqrMagnitude => x * x + y * y + z * z;
        public float magnitude => MathF.Sqrt(sqrMagnitude);
        public Vector3 normalized => magnitude > 1e-20f ? this / magnitude : zero;
        public void Normalize() { var n = normalized; x = n.x; y = n.y; z = n.z; }
        internal NV N => new(x, y, z);
        internal static Vector3 From(NV v) => new(v.X, v.Y, v.Z);
        public static float Dot(Vector3 a, Vector3 b) => a.x*b.x + a.y*b.y + a.z*b.z;
        public static Vector3 Cross(Vector3 a, Vector3 b) => From(NV.Cross(a.N, b.N));
        public static float SignedAngle(Vector3 from, Vector3 to, Vector3 axis)
        {
            var angle = MathF.Acos(Math.Clamp(Dot(from.normalized, to.normalized), -1f, 1f)) * Mathf.Rad2Deg;
            return angle * MathF.Sign(Dot(axis, Cross(from, to)));
        }
        public static Vector3 Slerp(Vector3 a, Vector3 b, float t)
        {
            float ma = a.magnitude, mb = b.magnitude;
            if (ma < 1e-20f || mb < 1e-20f) return a + (b-a)*t;
            var an = a / ma; var bn = b / mb;
            float omega = MathF.Acos(Math.Clamp(Dot(an,bn), -1f, 1f));
            if (omega < 1e-6f) return (a + (b-a)*t).normalized * (ma + (mb-ma)*t);
            float s = MathF.Sin(omega);
            var dir = an * (MathF.Sin((1-t)*omega)/s) + bn * (MathF.Sin(t*omega)/s);
            return dir * (ma + (mb-ma)*t);
        }
        public static Vector3 operator +(Vector3 a, Vector3 b) => new(a.x+b.x,a.y+b.y,a.z+b.z);
        public static Vector3 operator -(Vector3 a, Vector3 b) => new(a.x-b.x,a.y-b.y,a.z-b.z);
        public static Vector3 operator -(Vector3 a) => new(-a.x,-a.y,-a.z);
        public static Vector3 operator *(Vector3 a, float b) => new(a.x*b,a.y*b,a.z*b);
        public static Vector3 operator *(float b, Vector3 a) => a*b;
        public static Vector3 operator /(Vector3 a, float b) => new(a.x/b,a.y/b,a.z/b);
    }

    public struct Quaternion
    {
        public float x, y, z, w;
        public Quaternion(float x,float y,float z,float w){this.x=x;this.y=y;this.z=z;this.w=w;}
        public static Quaternion identity => new(0,0,0,1);
        internal NQ N => new(x,y,z,w);
        internal static Quaternion From(NQ q) => new(q.X,q.Y,q.Z,q.W);
        public static Quaternion operator *(Quaternion a, Quaternion b) => From(NQ.Multiply(a.N,b.N));
        public static Vector3 operator *(Quaternion q, Vector3 v) => Vector3.From(NV.Transform(v.N,q.N));
        public static Quaternion Inverse(Quaternion q) => From(NQ.Inverse(q.N));
        public static Quaternion Normalize(Quaternion q) => From(NQ.Normalize(q.N));
        public static Quaternion AngleAxis(float angle, Vector3 axis)
            => From(NQ.CreateFromAxisAngle(axis.normalized.N, angle * Mathf.Deg2Rad));
        public static Quaternion Euler(float x, float y, float z)
        {
            var qz = NQ.CreateFromAxisAngle(NV.UnitZ, z*Mathf.Deg2Rad);
            var qx = NQ.CreateFromAxisAngle(NV.UnitX, x*Mathf.Deg2Rad);
            var qy = NQ.CreateFromAxisAngle(NV.UnitY, y*Mathf.Deg2Rad);
            return From(NQ.Normalize(NQ.Multiply(qy, NQ.Multiply(qx,qz))));
        }
        public static Quaternion Euler(Vector3 e) => Euler(e.x,e.y,e.z);
        public static Quaternion LookRotation(Vector3 forward, Vector3 up)
        {
            var f=forward.normalized; var r=Vector3.Cross(up,f).normalized; var u=Vector3.Cross(f,r);
            var m = new System.Numerics.Matrix4x4(r.x,r.y,r.z,0,u.x,u.y,u.z,0,f.x,f.y,f.z,0,0,0,0,1);
            return From(NQ.CreateFromRotationMatrix(m));
        }
    }

    public static class Mathf
    {
        public const float PI = MathF.PI;
        public const float Deg2Rad = PI/180f;
        public const float Rad2Deg = 180f/PI;
        public static float Abs(float x)=>MathF.Abs(x);
        public static float Asin(float x)=>MathF.Asin(x);
        public static float Atan2(float y,float x)=>MathF.Atan2(y,x);
        public static float Clamp(float x,float a,float b)=>Math.Clamp(x,a,b);
        public static float Clamp01(float x)=>Math.Clamp(x,0f,1f);
        public static float Exp(float x)=>MathF.Exp(x);
        public static float InverseLerp(float a,float b,float x)
            => a==b?0f:Clamp01((x-a)/(b-a));
        public static float Lerp(float a,float b,float t)=>a+(b-a)*t;
        public static float Max(float a,float b)=>MathF.Max(a,b);
        public static int Max(int a,int b)=>Math.Max(a,b);
        public static float MoveTowards(float current,float target,float maxDelta)
            => MathF.Abs(target-current)<=maxDelta?target:current+MathF.Sign(target-current)*maxDelta;
        public static int RoundToInt(float x)=>(int)MathF.Round(x,MidpointRounding.AwayFromZero);
        public static float Repeat(float x,float length)=>x-MathF.Floor(x/length)*length;
        public static float Sin(float x)=>MathF.Sin(x);
        public static float Sqrt(float x)=>MathF.Sqrt(x);
    }
}
