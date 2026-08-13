using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using ProceduralWorld.Life;
using UnityEngine;

static string Sha(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
static float[] V(Vector3 v) => [v.x, v.y, v.z];
static float[] Q(Quaternion q) => [q.x, q.y, q.z, q.w];

static object Gene(PartGene p) => new
{
    type = p.type.ToString(),
    size = V(p.size),
    density = p.density,
    attach = V(p.attach),
    orient_deg = V(p.orient),
    mirror = p.mirror,
    port = p.port.ToString(),
    joint_amp_deg = p.jAmp,
    children = p.children.Select(Gene).ToArray(),
};

static PartGene Seg(
    Vector3 size,
    float density,
    Vector3 attach,
    float amp = 0f,
    PartType type = PartType.Segment,
    bool mirror = false,
    PortMode port = PortMode.None,
    Vector3? orient = null)
    => new()
    {
        type = type,
        size = size,
        density = density,
        attach = attach,
        orient = orient ?? Vector3.zero,
        mirror = mirror,
        port = port,
        jAmp = amp,
    };

static BodyGenome RootOnly()
    => new()
    {
        swimFreq = 2f,
        swimWave = 1f,
        root = Seg(new Vector3(0.5f, 0.35f, 0.7f), 4f, Vector3.zero, port: PortMode.Intake),
    };

static BodyGenome Swimmer()
{
    var g = new BodyGenome { swimFreq = 2f, swimWave = 1f };
    g.root = Seg(new Vector3(0.5f, 0.35f, 0.7f), 4f, Vector3.zero, port: PortMode.Intake);
    var a = Seg(new Vector3(0.4f, 0.30f, 0.6f), 4f, new Vector3(0, 0, 0.55f), 15f);
    var b = Seg(new Vector3(0.3f, 0.25f, 0.5f), 4f, new Vector3(0, 0, 0.50f), 25f);
    var c = Seg(
        new Vector3(0.7f, 0.9f, 0.4f),
        2f,
        new Vector3(0, 0, 0.45f),
        35f,
        PartType.Surface);
    b.children.Add(c);
    a.children.Add(b);
    g.root.children.Add(a);
    return g;
}

static BodyGenome Mirrored()
{
    var g = Swimmer();
    var fin = Seg(
        new Vector3(0.8f, 0.6f, 0.35f),
        2.3f,
        new Vector3(0.35f, 0.05f, 0.1f),
        12f,
        PartType.Surface,
        mirror: true,
        port: PortMode.Intake,
        orient: new Vector3(18f, -27f, 32f));
    g.root.children.Insert(0, fin);
    return g;
}

static BodyGenome DeepCap()
{
    var g = new BodyGenome { swimFreq = 1.3f, swimWave = 0.8f };
    g.root = Seg(new Vector3(0.5f, 0.3f, 0.7f), 4f, Vector3.zero, port: PortMode.Intake);
    var current = g.root;
    for (var depth = 1; depth <= 9; depth++)
    {
        var child = Seg(
            new Vector3(0.35f, 0.24f, 0.5f),
            3.5f + 0.1f * depth,
            new Vector3(0.03f * (depth % 2), 0, 0.45f),
            8f + 4f * depth,
            depth == 5 ? PartType.Surface : PartType.Segment,
            mirror: depth == 2);
        current.children.Add(child);
        current = child;
    }
    return g;
}

static BodyGenome Wide16()
{
    var g = new BodyGenome { swimFreq = 1.6f, swimWave = 1.15f };
    g.root = Seg(new Vector3(0.55f, 0.38f, 0.8f), 4.2f, Vector3.zero, port: PortMode.Intake);
    for (var i = 0; i < 15; i++)
    {
        var angle = 2f * MathF.PI * i / 15f;
        g.root.children.Add(Seg(
            new Vector3(0.18f + 0.02f * (i % 4), 0.14f + 0.01f * (i % 3), 0.32f + 0.025f * i),
            2.0f + 0.25f * (i % 5),
            new Vector3(0.32f * MathF.Sin(angle), 0.06f * ((i % 3) - 1), 0.18f + 0.07f * i),
            5f + 2f * i,
            i % 4 == 0 ? PartType.Surface : PartType.Segment,
            orient: new Vector3(7f * (i % 3), -18f + 3f * i, 11f - 2f * i)));
    }
    return g;
}

static object Trace(BodyGenome g, string mode, int steps)
{
    var segs = SwimEval.Reconstruct(g, out int tail);
    SwimEval.CenterOfMass(segs, out float mass);
    var sim = SwimEval.Sim.CreateLive(segs, tail, mass, g.swimFreq, Vector3.forward, Vector3.zero);
    if (mode == "turn_pos") sim.SetTurnCmd(1f);
    if (mode == "turn_neg") sim.SetTurnCmd(-1f);
    if (mode == "homing") sim.SetDesiredHeading(Vector3.right);
    var flags = BindingFlags.Instance | BindingFlags.NonPublic;
    var lyaw = typeof(SwimEval.Sim).GetField("_Lyaw", flags)!;
    var turn = typeof(SwimEval.Sim).GetField("_turnCmd", flags)!;
    var rows = new List<object>();
    for (var step = 0; step < steps; step++)
    {
        if (mode == "homing" && step % 2 == 0) sim.LatchTurnCmd();
        sim.StepLive();
        rows.Add(new
        {
            step = step + 1,
            position = V(sim.ComNow()),
            velocity = V(sim.Velocity),
            orientation = Q(sim.OrientationNow()),
            omega_yaw = sim.OmegaNow(),
            yaw_momentum = (float)lyaw.GetValue(sim)!,
            turn_cmd = (float)turn.GetValue(sim)!,
            live_work = sim.LiveWork,
        });
    }
    return new { mode, rows };
}

var cases = new List<(string id, BodyGenome genome)>
{
    ("root-only", RootOnly()),
    ("swimmer", Swimmer()),
    ("mirrored", Mirrored()),
    ("deep-cap", DeepCap()),
    ("wide-16", Wide16()),
};
for (var seed = 0; seed < 27; seed++)
    cases.Add(($"random-{seed:D2}", BodyGenome.Random(new Random(0x5A17 + seed))));

var bodies = new List<object>();
foreach (var (id, genome) in cases)
{
    var measure = genome.Measure();
    var segments = SwimEval.ReconstructForTest(genome, out int tail);
    bodies.Add(new
    {
        id,
        genotype = new { swim_freq_hz = genome.swimFreq, swim_wave_rad_per_depth = genome.swimWave, root = Gene(genome.root) },
        donor_measure = new
        {
            measure.mass,
            measure.area,
            measure.intake,
            measure.bulk,
            measure.avgDensity,
            measure.length,
            measure.girth,
            measure.fineness,
            measure.propArea,
            measure.asymmetry,
            measure.compactness,
            measure.swimProxy,
            parts = genome.PartCount(),
        },
        tail,
        segments = segments.Select((s, index) => new
        {
            slot = index,
            rest_pos = V(s.restPos),
            rest_rot = Q(s.restRot),
            local_pos = V(s.localPos),
            local_rot = Q(s.localRot),
            abc = new[] { s.a, s.b, s.c },
            s.volume,
            mass_sim = s.mass,
            drag_area = new[] { s.areaX, s.areaY, s.areaZ },
            added_mass_kg = new[] { s.maX, s.maY, s.maZ },
            s.finMaPerp,
            s.parentIndex,
            s.depth,
            s.side,
            s.ampDeg,
            s.phase,
            s.hasJoint,
            s.isTail,
            type = s.gene.type.ToString(),
            port = s.gene.port.ToString(),
        }).ToArray(),
        traces = id == "swimmer"
            ? new[] { Trace(genome, "straight", 120), Trace(genome, "turn_pos", 120), Trace(genome, "turn_neg", 120), Trace(genome, "homing", 240) }
            : Array.Empty<object>(),
    });
}

var root = Directory.GetCurrentDirectory();
var bodyGraphPath = "/mnt/c/Users/cddal/game prototype/Assets/ProceduralWorld/Scripts/Life/BodyGraph.cs";
var swimEvalPath = "/mnt/c/Users/cddal/game prototype/Assets/ProceduralWorld/Scripts/Life/SwimEval.cs";
var mutationPath = "/mnt/c/Users/cddal/game prototype/Assets/ProceduralWorld/Scripts/Life/MutationNoise.cs";
var output = new
{
    schema = "sirrobin.development-live.donor.v1",
    donor_sources = new[]
    {
        new { path = bodyGraphPath, sha256 = Sha(bodyGraphPath) },
        new { path = swimEvalPath, sha256 = Sha(swimEvalPath) },
        new { path = mutationPath, sha256 = Sha(mutationPath) },
    },
    frame = "donor Unity: x-right, y-up, z-longitudinal; StepLive travels +z",
    bodies,
};
var outputPath = Path.Combine(root, "oracle", "fixtures", "live", "donor_development_live.json");
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
File.WriteAllText(outputPath, JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"wrote {outputPath} ({bodies.Count} bodies)");
