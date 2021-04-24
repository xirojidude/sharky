using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDKBase;
using VRC.Udon;
using VRC.SDK3.Video.Components;
using VRC.SDK3.Video.Components.AVPro;
using VRC.SDK3.Video.Components.Base;
using VRC.SDK3.Components.Video;

public class SkyController : UdonSharpBehaviour
{
    public int _currentSky=0;

    [UdonSynced]
    public int _syncedSky=0;

//    public int _queuedSky=0;
    private float _lastSkyTime=0.0f;
    private bool _skyQueued=false;
    public float SkyChangeRate=5.0f;


    [Tooltip("This list of skyboxes for the world")]
    public Material[] skybox;

    [Tooltip("This list of videos plays to match theskybox")]
    public VRCUrl[] playlist;

    public VRCAVProVideoPlayer videoPlayer;
//    public VRCAVProVideoPlayer musicPlayer;
    public VRCUnityVideoPlayer musicPlayer;
//    public VRCUrl  soundUrl;


    public GameObject Sun;
    public GameObject Moon;

    public GameObject masterCheckObj;

    public Text ownerTextField;
    public Text masterTextField;

    public Toggle globalCheckbox;
    bool _globalSky = true;

/*
0 blizzard
1 spaceship
2 space
3 windydesert
4 spacestation
5 caves
6 swamp
7 springbirds
8 crickets
9 hauntedspace
10 ocean
11 interstallar
12 cavedrips
13 canyon
14 nuclearwinter
15 desertcanyon
16 eerieazathoth
17 spacesignals
18 bio
19 music
20 deathstar
*/

    private int[] soundMap = new int[] {0,16,2,10,9,16  ,0,17,20,3,14,20  ,2,20,8,9,15,3  ,0,2,12,17,15,3  ,3,7,0,5,11,5  ,4,15,19,19,19,19  ,19,15,18,18,16,19  ,0,14,3,19,19,16   ,0,14,3,19,19,16    ,0,14,3,19,19,16    ,0,14,3,19,19,16};
    private int[] sunMap = new int[]   {1,1,1,1,1,0     ,1,1,1,1,0,1      ,1,1,2,1,1,1    ,1,0,0,0,1,1     ,1,1,1,0,0,0   ,0,1,0,0,0,0       ,0,1,0,0,0,0        ,0,0,1,0,0,1       ,0,0,1,0,0,1        ,0,0,1,0,0,1        ,0,0,1,0,0,1};
    private int[] musicMap = new int[] {0,0,0,0,0,0     ,0,0,0,0,0,0      ,0,0,0,0,0,0    ,0,0,0,0,0,0     ,0,0,0,0,0,0   ,0,0,1,1,1,1       ,1,0,0,0,0,1        ,0,0,0,1,1,0       ,0,0,0,1,1,0        ,0,0,0,1,1,0        ,0,0,0,1,1,0};

//    private string[] SkyParams  = {"Panoskybox, Mountain Clouds,  TER, 1, 0, 0, wind, 1"}; //new string[1];
//    private string[] SkyParamsMap = new string[48];
    
            // shader         label              cat gpu ground sun video enabled
//    SkyParams[0] = "Panoskybox, Mountain Clouds,  TER, 1, 0, 0, wind, 1";
/*
    SkyParamsMap = {
    "PanoSkybox",
    "FoggyMountain",
    "StarNestSkybox",
    "Ocean",
    "MarsJetPack",
    "CanyonPass",

    "tinyCloud",
    "FoggyMountainGreen1",
    "PanoSkybox3",
    "AbstractTerrainObjects",
    "Xann",
    "DeathStar",

    "Stars",
    "SkyboxMinimal",
    "Aurora",
    "PlanetGreen",
    "DesertPassage",
    "DesertSand",

    "PlanetMars",
    "FractalStarfield",
    "Fissure",
    "Postcard",
    "DesertCanyon",
    "MountainPath",
    
    "Terrainz",
    "GrassyHillis",
    "PlanetOther",
    "Fissure",
    "Gargantuan",
    "LavaPlanet",

    "LunarDebris",
    "RockyGorge",
    "ProceduralTechRings",
    "RainboxCave",
    "JetStream",
    "JaszUniverse",

    "SimplicityGalaxy",
    "DesertCanyon",
    "Virus",
    "Biomine",
    "StarNursery",
    "Doodling",

    "SkinPeeler",
    "SunSurface",
    "CombustibleClouds",
    "TransparentIsoceles",
    "VenusBebop",
    "NeptuneRacing"};

*/

//SkyRootObject shaderList = new SkyRootObject();
//shaderList.items = new List<SkyItem> {};
/*
new SkyItem {id=	4	, shader=	"Alps"	, label=	"Alps"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	5	, shader=	"Apollonian"	, label=	"Apollonian"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"eerie"	, enabled=	1	},
new SkyItem {id=	6	, shader=	"AsteroidAbduction"	, label=	"AsteroidAbduction"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"eerie"	, enabled=	1	},
new SkyItem {id=	7	, shader=	"AsteroidDebris"	, label=	"AsteroidDebris"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	0	, video=	"space"	, enabled=	1	},
new SkyItem {id=	8	, shader=	"Aurora"	, label=	"Aurora"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	2	, video=	"crickets"	, enabled=	1	},
new SkyItem {id=	9	, shader=	"Aurora2"	, label=	"Aurora2"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	2	, video=	"crickets"	, enabled=	1	},
new SkyItem {id=	10	, shader=	"BigBang"	, label=	"BigBang"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	11	, shader=	"Biomine"	, label=	"Biomine"	, cat=	"BIO"	, gpu=	1	, ground=	0	, sun=	0	, video=	"bio"	, enabled=	1	},
new SkyItem {id=	12	, shader=	"CanyonPass"	, label=	"CanyonPass"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	},
new SkyItem {id=	13	, shader=	"CheapCloud"	, label=	"CheapCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	14	, shader=	"CaveDweller"	, label=	"CaveDweller"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	},
new SkyItem {id=	15	, shader=	"CheaperCloud"	, label=	"CheaperCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	16	, shader=	"CityFlight"	, label=	"CityFlight"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	2	, video=	"future"	, enabled=	1	},
new SkyItem {id=	17	, shader=	"CloudQuad"	, label=	"CloudQuad"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	18	, shader=	"CombustibleCloud"	, label=	"CombustibleCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	19	, shader=	"Cook19"	, label=	"Cook19"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"snow"	, enabled=	1	},
new SkyItem {id=	20	, shader=	"CubeScape"	, label=	"CubeScape"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	21	, shader=	"DeathStar"	, label=	"DeathStar"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"starwars"	, enabled=	1	},
new SkyItem {id=	22	, shader=	"DesertCanyon"	, label=	"DesertCanyon"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	},
new SkyItem {id=	23	, shader=	"DesertCanyon1"	, label=	"DesertCanyon1"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	},
new SkyItem {id=	24	, shader=	"DesertPassage"	, label=	"DesertPassage"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	},
new SkyItem {id=	25	, shader=	"DesertSand"	, label=	"DesertSand"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"desert"	, enabled=	1	},
new SkyItem {id=	26	, shader=	"Snow Tunnel"	, label=	"Snow Tunnel"	, cat=	"TUN"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	27	, shader=	"Dynamism"	, label=	"Dynamism"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"multibuffer"	, enabled=	1	},
new SkyItem {id=	28	, shader=	"Escalated"	, label=	"Escalated"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"mountains"	, enabled=	1	},
new SkyItem {id=	29	, shader=	"Fissure"	, label=	"Fissure"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	},
new SkyItem {id=	30	, shader=	"Fissure1"	, label=	"Fissure1"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	},
new SkyItem {id=	31	, shader=	"FoggyMountain"	, label=	"FoggyMountain"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	},
new SkyItem {id=	32	, shader=	"FoggyMountain1"	, label=	"FoggyMountain1"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	},
new SkyItem {id=	33	, shader=	"Fractal32"	, label=	"Fractal32"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	34	, shader=	"FractalStarfield"	, label=	"FractalStarfield"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	1	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	35	, shader=	"GalaxyNavigator"	, label=	"GalaxyNavigator"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	36	, shader=	"Gargantuan"	, label=	"Gargantuan"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"interstellar"	, enabled=	1	},
new SkyItem {id=	37	, shader=	"Globules"	, label=	"Globules"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"bio"	, enabled=	1	},
new SkyItem {id=	38	, shader=	"GrassyHills"	, label=	"GrassyHills"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"birds"	, enabled=	1	},
new SkyItem {id=	39	, shader=	"Iceberg"	, label=	"Iceberg"	, cat=	"WTR"	, gpu=	1	, ground=	0	, sun=	1	, video=	"water"	, enabled=	1	},
new SkyItem {id=	40	, shader=	"IceMountains"	, label=	"IceMountains"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	},
new SkyItem {id=	41	, shader=	"JaszUniverse"	, label=	"JaszUniverse"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	42	, shader=	"Jellyfish"	, label=	"Jellyfish"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"water"	, enabled=	1	},
new SkyItem {id=	43	, shader=	"JetStream"	, label=	"JetStream"	, cat=	"TUN"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	44	, shader=	"LavaPlanet"	, label=	"LavaPlanet"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	},
new SkyItem {id=	45	, shader=	"LunarDebris"	, label=	"LunarDebris"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	0	, video=	"space"	, enabled=	1	},
new SkyItem {id=	46	, shader=	"MarsJetpack"	, label=	"MarsJetpack"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	47	, shader=	"Minimal"	, label=	"Minimal"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	1	, video=	"eerie"	, enabled=	1	},
new SkyItem {id=	48	, shader=	"MountainPath"	, label=	"MountainPath"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	},
new SkyItem {id=	49	, shader=	"MysteryMountain"	, label=	"MysteryMountain"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	},
new SkyItem {id=	50	, shader=	"NebulaPlexus"	, label=	"NebulaPlexus"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	51	, shader=	"NeptuneRacing"	, label=	"NeptuneRacing"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	52	, shader=	"Ocean"	, label=	"Ocean"	, cat=	"WTR"	, gpu=	1	, ground=	0	, sun=	1	, video=	"water"	, enabled=	1	},
new SkyItem {id=	53	, shader=	"Panoskybox"	, label=	"Panoskybox"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	54	, shader=	"Panoskybox"	, label=	"Panoskybox"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	55	, shader=	"Panoskybox"	, label=	"Panoskybox"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	56	, shader=	"Panosphere"	, label=	"Panosphere"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	57	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	58	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	59	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	60	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	61	, shader=	"PostCard"	, label=	"PostCard"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	62	, shader=	"ProceduralTechRings"	, label=	"ProceduralTechRings"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	63	, shader=	"RainbowCave"	, label=	"RainbowCave"	, cat=	"TUN"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	64	, shader=	"RockyGorge"	, label=	"RockyGorge"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	},
new SkyItem {id=	65	, shader=	"RollingHills"	, label=	"RollingHills"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"birds"	, enabled=	1	},
new SkyItem {id=	66	, shader=	"ShatteredDojo"	, label=	"ShatteredDojo"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"eerie"	, enabled=	1	},
new SkyItem {id=	67	, shader=	"SimplicityGalaxy"	, label=	"SimplicityGalaxy"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	68	, shader=	"SkinPeeler"	, label=	"SkinPeeler"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	69	, shader=	"SkydomeTwinSuns"	, label=	"SkydomeTwinSuns"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	70	, shader=	"SmallPlanet"	, label=	"SmallPlanet"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	71	, shader=	"StarfieldDarkDust"	, label=	"StarfieldDarkDust"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	72	, shader=	"Starnest"	, label=	"Starnest"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	73	, shader=	"StarNursery"	, label=	"StarNursery"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	74	, shader=	"Stars"	, label=	"Stars"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	},
new SkyItem {id=	75	, shader=	"Storm"	, label=	"Storm"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	76	, shader=	"SunSurface"	, label=	"SunSurface"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	0	, video=	"eruption"	, enabled=	1	},
new SkyItem {id=	77	, shader=	"TerrainPolyhedron"	, label=	"TerrainPolyhedron"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"desert"	, enabled=	1	},
new SkyItem {id=	78	, shader=	"Terrainz"	, label=	"Terrainz"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	},
new SkyItem {id=	79	, shader=	"TinyCloud"	, label=	"TinyCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	80	, shader=	"TransparentIsoceles"	, label=	"TransparentIsoceles"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"crystal"	, enabled=	1	},
new SkyItem {id=	81	, shader=	"VenusBebop"	, label=	"VenusBebop"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	0	, video=	"desert"	, enabled=	1	},
new SkyItem {id=	82	, shader=	"Virus"	, label=	"Virus"	, cat=	"BIO"	, gpu=	1	, ground=	0	, sun=	0	, video=	"bio"	, enabled=	1	},
new SkyItem {id=	83	, shader=	"VolumetricLines"	, label=	"VolumetricLines"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	1	, video=	"techno"	, enabled=	1	},
new SkyItem {id=	84	, shader=	"Weather"	, label=	"Weather"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	},
new SkyItem {id=	85	, shader=	"Xyptonjtroz"	, label=	"Xyptonjtroz"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}
};
*/
  void Start()
    {
#if !UNITY_EDITOR // Causes null ref exceptions so just exclude it from the editor
        masterTextField.text = "Master: "+ Networking.GetOwner(masterCheckObj).displayName;
#endif   
    
#if !UNITY_EDITOR // Causes null ref exceptions so just exclude it from the editor
        ownerTextField.text = "Owner: " + Networking.GetOwner(gameObject).displayName;
#endif
 
    }

    // wish I could get a reference to button that was clicked so I didn't have to do this hack
    public void ShowSky0() { showSky(0); }
    public void ShowSky1() { showSky(1); }
    public void ShowSky2() { showSky(2); }
    public void ShowSky3() { showSky(3); }
    public void ShowSky4() { showSky(4); }
    public void ShowSky5() { showSky(5); }
    public void ShowSky6() { showSky(6); }
    public void ShowSky7() { showSky(7); }
    public void ShowSky8() { showSky(8); }
    public void ShowSky9() { showSky(9); }

    public void ShowSky10() { showSky(10); }
    public void ShowSky11() { showSky(11); }
    public void ShowSky12() { showSky(12); }
    public void ShowSky13() { showSky(13); }
    public void ShowSky14() { showSky(14); }
    public void ShowSky15() { showSky(15); }
    public void ShowSky16() { showSky(16); }
    public void ShowSky17() { showSky(17); }
    public void ShowSky18() { showSky(18); }
    public void ShowSky19() { showSky(19); }

    public void ShowSky20() { showSky(20); }
    public void ShowSky21() { showSky(21); }
    public void ShowSky22() { showSky(22); }
    public void ShowSky23() { showSky(23); }
    public void ShowSky24() { showSky(24); }
    public void ShowSky25() { showSky(25); }
    public void ShowSky26() { showSky(26); }
    public void ShowSky27() { showSky(27); }
    public void ShowSky28() { showSky(28); }
    public void ShowSky29() { showSky(29); }

    public void ShowSky30() { showSky(30); }
    public void ShowSky31() { showSky(31); }
    public void ShowSky32() { showSky(32); }
    public void ShowSky33() { showSky(33); }
    public void ShowSky34() { showSky(34); }
    public void ShowSky35() { showSky(35); }
    public void ShowSky36() { showSky(36); }
    public void ShowSky37() { showSky(37); }
    public void ShowSky38() { showSky(38); }
    public void ShowSky39() { showSky(39); }

    public void ShowSky40() { showSky(40); }
    public void ShowSky41() { showSky(41); }
    public void ShowSky42() { showSky(42); }
    public void ShowSky43() { showSky(43); }
    public void ShowSky44() { showSky(44); }
    public void ShowSky45() { showSky(45); }
    public void ShowSky46() { showSky(46); }
    public void ShowSky47() { showSky(47); }
    public void ShowSky48() { showSky(48); }
    public void ShowSky49() { showSky(49); }

    public void ShowSky50() { showSky(50); }
    public void ShowSky51() { showSky(51); }
    public void ShowSky52() { showSky(52); }
    public void ShowSky53() { showSky(53); }
    public void ShowSky54() { showSky(54); }
    public void ShowSky55() { showSky(55); }
    public void ShowSky56() { showSky(56); }
    public void ShowSky57() { showSky(57); }
    public void ShowSky58() { showSky(58); }
    public void ShowSky59() { showSky(59); }

    public void ShowSky60() { showSky(60); }
    public void ShowSky61() { showSky(61); }
    public void ShowSky62() { showSky(62); }
    public void ShowSky63() { showSky(63); }
    public void ShowSky64() { showSky(64); }
    public void ShowSky65() { showSky(65); }
    public void ShowSky66() { showSky(66); }
    public void ShowSky67() { showSky(67); }
    public void ShowSky68() { showSky(68); }
    public void ShowSky69() { showSky(69); }

    public void NextSky() 
    {
        Debug.Log("******************  NextSky Triggered  ***********************");

        _currentSky++;
        showSky(_currentSky);
    }

    public void PreviousSky()
    {
        Debug.Log("******************  PrevSky Triggered  ***********************");

        _currentSky--;
        showSky(_currentSky);
    }

    private void showSky(int id) 
    {
        _currentSky = id;
        if (_currentSky>=skybox.Length) _currentSky = 0;
        if (_currentSky<0) _currentSky = skybox.Length-1;

        RenderSettings.skybox = skybox[_currentSky];
        Debug.Log("NEW SKY::::::::::::::::" + skybox[_currentSky].ToString());

        if (Time.time - _lastSkyTime > SkyChangeRate) 
        {
            _lastSkyTime = Time.time;
            _skyQueued = false;
            // if (musicMap[_currentSky] == 1)
            // {
                if (musicPlayer!=null) {
                    musicPlayer.LoadURL( playlist[ soundMap[_currentSky] ] );
                }
                // if (videoPlayer!=null) {
                //     videoPlayer.Stop();
                // }
            // }
            // else
            // {
            //     if (videoPlayer!=null) {
            //         videoPlayer.LoadURL( playlist[ soundMap[_currentSky] ] );
            //     }
            //     if (musicPlayer!=null) {
            //         musicPlayer.Stop();
            //     }
            // }
        }
        else
        {
            _skyQueued = true;
        }
        if (Sun!=null) {
            if (sunMap[_currentSky] == 1) 
                Sun.SetActive(true);
            else
                Sun.SetActive(false);
        }


        if (_globalSky) {
  //          if (Networking.IsOwner(gameObject)) 
                _syncedSky = _currentSky;
        }
    }

    void Update()
    {
        if (_skyQueued)
        {
            if (Time.time - _lastSkyTime > SkyChangeRate) 
            {
                showSky(_currentSky);
            }
        }
    }

    public void TakeOwnership()
    {
        Debug.Log("Take Ownership ");
        Debug.Log("From "+Networking.GetOwner(gameObject).displayName + " and assign it to "+Networking.LocalPlayer.displayName);
   //     if (Networking.IsMaster || !_masterOnly)
   //     {
            if (!Networking.IsOwner(gameObject))
            {
                Networking.SetOwner(Networking.LocalPlayer, gameObject);
            }
#if !UNITY_EDITOR // Causes null ref exceptions so just exclude it from the editor
            ownerTextField.text = "Owner: " + Networking.GetOwner(gameObject).displayName;
#endif
//     }
    }

    public void ToggleGlobal()
    {
        _globalSky =  globalCheckbox.isOn;
        if (_globalSky) 
        {
            _currentSky = _syncedSky;
            showSky(_currentSky);
        }
    }

    public void OnDeserialization()
    {
        if (_syncedSky != _currentSky)
        {
            if (_globalSky)
            {
                _currentSky = _syncedSky;
                showSky(_currentSky);
            }
        }
    }

}
