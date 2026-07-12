using System.Collections.Generic;
using UnityEngine;

namespace ProceduralWorld.Life
{
    public enum PartType { Segment, Surface }

    public sealed class PartGene
    {
        public PartType type = PartType.Segment;
        public Vector3 size;
        public float density;
        public Vector3 attach;
        public Vector3 orient;
        public bool mirror;
        public float jAmp;
        public readonly List<PartGene> children = new();
    }

    public sealed class BodyGenome
    {
        public PartGene root = new();
        public float swimFreq;
        public float swimWave;
    }
}

