using System.Security.Cryptography;
using System.Text.Json;
using System.Reflection;
using ProceduralWorld.Life;
using UnityEngine;

static string Sha(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
static Vector3 V(JsonElement e) => new(e[0].GetSingle(), e[1].GetSingle(), e[2].GetSingle());

static BodyGenome Genome(JsonElement body)
{
    var segments = body.GetProperty("segments").EnumerateArray().ToArray();
    var genes = new PartGene[segments.Length + 1];
    foreach (var seg in segments)
    {
        int slot = seg.GetProperty("slot").GetInt32();
        var abc = V(seg.GetProperty("abc_m"));
        bool surface = seg.GetProperty("is_surface").GetBoolean();
        float span = seg.GetProperty("fin_span_m").GetSingle();
        genes[slot] = new PartGene {
            type = surface ? PartType.Surface : PartType.Segment,
            size = new Vector3(2*abc.x, surface ? span : 2*abc.y, 2*abc.z),
            density = seg.GetProperty("density_gene_sim_mass_m3").GetSingle(),
            attach = V(seg.GetProperty("local_pos_m")),
            orient = V(seg.GetProperty("local_euler_deg_xyz")),
            mirror = false,
            jAmp = seg.GetProperty("amp_deg").GetSingle(),
        };
    }
    foreach (var seg in segments)
    {
        int slot=seg.GetProperty("slot").GetInt32(), parent=seg.GetProperty("parent").GetInt32();
        if(parent>0) genes[parent].children.Add(genes[slot]);
    }
    return new BodyGenome {
        root=genes[1],
        swimFreq=body.GetProperty("swim_freq_hz").GetSingle(),
        swimWave=body.GetProperty("swim_wave_rad_per_depth").GetSingle(),
    };
}

var root = Directory.GetCurrentDirectory();
var corpusPath = Path.Combine(root,"oracle","fixtures","corpus.json");
var outputPath = Path.Combine(root,"oracle","fixtures","gain0_donor.json");
using var doc = JsonDocument.Parse(File.ReadAllText(corpusPath));
var output = new List<object>();
foreach(var body in doc.RootElement.GetProperty("bodies").EnumerateArray())
{
    var g=Genome(body);
    var segs=SwimEval.ReconstructForTest(g,out int tail);
    var segList=segs.ToList();
    Vector3 comRest=Vector3.zero; float mBody=0;
    foreach(var s in segs){mBody+=s.mass;comRest+=s.mass*s.restPos;}
    comRest=mBody>1e-6f?comRest/mBody:Vector3.zero;
    var fHat=comRest-segs[tail].restPos;fHat.y=0;
    if(fHat.sqrMagnitude<1e-8f)fHat=Vector3.back;fHat.Normalize();
    var sim=new SwimEval.Sim(segList,tail,mBody,g.swimFreq,fHat,0.1f,Vector3.zero,0f);
    var flags=BindingFlags.Instance|BindingFlags.NonPublic;
    var vField=typeof(SwimEval.Sim).GetField("_vCom",flags)!;
    var xField=typeof(SwimEval.Sim).GetField("_xCom",flags)!;
    var trace=new List<object>();
    for(int step=0;step<32;step++){
        sim.Step(false);var v=(Vector3)vField.GetValue(sim)!;var x=(Vector3)xField.GetValue(sim)!;var com=sim.ComNow();
        trace.Add(new{step=step+1,v_com=new[]{v.x,v.y,v.z},x_origin=new[]{x.x,x.y,x.z},com_world=new[]{com.x,com.y,com.z}});
    }
    var perf=SwimEval.Evaluate(g);
    var diag=SwimEval.EvaluateDiag(g);
    output.Add(new {
        id=body.GetProperty("id").GetString(), tail,
        segments=segs.Select((s,i)=>new {slot=i+1,rest_pos=new[]{s.restPos.x,s.restPos.y,s.restPos.z},
            rest_rot=new[]{s.restRot.x,s.restRot.y,s.restRot.z,s.restRot.w},abc=new[]{s.a,s.b,s.c},
            mass_sim=s.mass,added_mass_kg=new[]{s.maX,s.maY,s.maZ},parent=s.parentIndex+1,depth=s.depth}).ToArray(),
        trace,
        aggregate=new {perf.cruiseSpeed,perf.costOfTransport,perf.reactiveRatio,diag.mechWork,diag.dragImpulse,diag.measureTime}
    });
}
var donorPath="/mnt/c/Users/cddal/game prototype/Assets/ProceduralWorld/Scripts/Life/SwimEval.cs";
var payload=new {schema="sirrobin.locomotion.gain0.v1", corpus_sha256=Sha(corpusPath), donor_path=donorPath,
    donor_sha256=Sha(donorPath), generated_utc=DateTimeOffset.UtcNow, bodies=output};
File.WriteAllText(outputPath,JsonSerializer.Serialize(payload,new JsonSerializerOptions{WriteIndented=true})+"\n");
Console.WriteLine($"wrote {outputPath} ({output.Count} bodies)");
