
using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using VRC.SDK3.Components;
using VRC.SDK3.Video.Components;
using VRC.SDK3.Video.Components.AVPro;
using VRC.SDK3.Video.Components.Base;

using VRC.SDK3.Components.Video;
using VRC.SDKBase;
using VRC.Udon;

/*
var shaderList = new RootObject();
shaderList.items = new List<Item> 
                          {

new Item {id=	4	, shader=	"Alps"	, label=	"Alps"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	5	, shader=	"Apollonian"	, label=	"Apollonian"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"eerie"	, enabled=	1	}.
new Item {id=	6	, shader=	"AsteroidAbduction"	, label=	"AsteroidAbduction"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"eerie"	, enabled=	1	}.
new Item {id=	7	, shader=	"AsteroidDebris"	, label=	"AsteroidDebris"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	0	, video=	"space"	, enabled=	1	}.
new Item {id=	8	, shader=	"Aurora"	, label=	"Aurora"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	2	, video=	"crickets"	, enabled=	1	}.
new Item {id=	9	, shader=	"Aurora2"	, label=	"Aurora2"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	2	, video=	"crickets"	, enabled=	1	}.
new Item {id=	10	, shader=	"BigBang"	, label=	"BigBang"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	11	, shader=	"Biomine"	, label=	"Biomine"	, cat=	"BIO"	, gpu=	1	, ground=	0	, sun=	0	, video=	"bio"	, enabled=	1	}.
new Item {id=	12	, shader=	"CanyonPass"	, label=	"CanyonPass"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	}.
new Item {id=	13	, shader=	"CheapCloud"	, label=	"CheapCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	14	, shader=	"CaveDweller"	, label=	"CaveDweller"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	}.
new Item {id=	15	, shader=	"CheaperCloud"	, label=	"CheaperCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	16	, shader=	"CityFlight"	, label=	"CityFlight"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	2	, video=	"future"	, enabled=	1	}.
new Item {id=	17	, shader=	"CloudQuad"	, label=	"CloudQuad"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	18	, shader=	"CombustibleCloud"	, label=	"CombustibleCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	19	, shader=	"Cook19"	, label=	"Cook19"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"snow"	, enabled=	1	}.
new Item {id=	20	, shader=	"CubeScape"	, label=	"CubeScape"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	21	, shader=	"DeathStar"	, label=	"DeathStar"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"starwars"	, enabled=	1	}.
new Item {id=	22	, shader=	"DesertCanyon"	, label=	"DesertCanyon"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	}.
new Item {id=	23	, shader=	"DesertCanyon1"	, label=	"DesertCanyon1"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	}.
new Item {id=	24	, shader=	"DesertPassage"	, label=	"DesertPassage"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	}.
new Item {id=	25	, shader=	"DesertSand"	, label=	"DesertSand"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"desert"	, enabled=	1	}.
new Item {id=	26	, shader=	"Snow Tunnel"	, label=	"Snow Tunnel"	, cat=	"TUN"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	27	, shader=	"Dynamism"	, label=	"Dynamism"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"multibuffer"	, enabled=	1	}.
new Item {id=	28	, shader=	"Escalated"	, label=	"Escalated"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"mountains"	, enabled=	1	}.
new Item {id=	29	, shader=	"Fissure"	, label=	"Fissure"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	}.
new Item {id=	30	, shader=	"Fissure1"	, label=	"Fissure1"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	}.
new Item {id=	31	, shader=	"FoggyMountain"	, label=	"FoggyMountain"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	}.
new Item {id=	32	, shader=	"FoggyMountain1"	, label=	"FoggyMountain1"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	}.
new Item {id=	33	, shader=	"Fractal32"	, label=	"Fractal32"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	34	, shader=	"FractalStarfield"	, label=	"FractalStarfield"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	1	, video=	"techno"	, enabled=	1	}.
new Item {id=	35	, shader=	"GalaxyNavigator"	, label=	"GalaxyNavigator"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	36	, shader=	"Gargantuan"	, label=	"Gargantuan"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"interstellar"	, enabled=	1	}.
new Item {id=	37	, shader=	"Globules"	, label=	"Globules"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"bio"	, enabled=	1	}.
new Item {id=	38	, shader=	"GrassyHills"	, label=	"GrassyHills"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"birds"	, enabled=	1	}.
new Item {id=	39	, shader=	"Iceberg"	, label=	"Iceberg"	, cat=	"WTR"	, gpu=	1	, ground=	0	, sun=	1	, video=	"water"	, enabled=	1	}.
new Item {id=	40	, shader=	"IceMountains"	, label=	"IceMountains"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	}.
new Item {id=	41	, shader=	"JaszUniverse"	, label=	"JaszUniverse"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	42	, shader=	"Jellyfish"	, label=	"Jellyfish"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"water"	, enabled=	1	}.
new Item {id=	43	, shader=	"JetStream"	, label=	"JetStream"	, cat=	"TUN"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	44	, shader=	"LavaPlanet"	, label=	"LavaPlanet"	, cat=	"CAV"	, gpu=	1	, ground=	0	, sun=	0	, video=	"cave"	, enabled=	1	}.
new Item {id=	45	, shader=	"LunarDebris"	, label=	"LunarDebris"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	0	, video=	"space"	, enabled=	1	}.
new Item {id=	46	, shader=	"MarsJetpack"	, label=	"MarsJetpack"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	47	, shader=	"Minimal"	, label=	"Minimal"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	1	, video=	"eerie"	, enabled=	1	}.
new Item {id=	48	, shader=	"MountainPath"	, label=	"MountainPath"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	}.
new Item {id=	49	, shader=	"MysteryMountain"	, label=	"MysteryMountain"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	}.
new Item {id=	50	, shader=	"NebulaPlexus"	, label=	"NebulaPlexus"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	51	, shader=	"NeptuneRacing"	, label=	"NeptuneRacing"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	52	, shader=	"Ocean"	, label=	"Ocean"	, cat=	"WTR"	, gpu=	1	, ground=	0	, sun=	1	, video=	"water"	, enabled=	1	}.
new Item {id=	53	, shader=	"Panoskybox"	, label=	"Panoskybox"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	54	, shader=	"Panoskybox"	, label=	"Panoskybox"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	55	, shader=	"Panoskybox"	, label=	"Panoskybox"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	56	, shader=	"Panosphere"	, label=	"Panosphere"	, cat=	"PNO"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	57	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	58	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	59	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	60	, shader=	"PlanetMars"	, label=	"PlanetMars"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	61	, shader=	"PostCard"	, label=	"PostCard"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	62	, shader=	"ProceduralTechRings"	, label=	"ProceduralTechRings"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	63	, shader=	"RainbowCave"	, label=	"RainbowCave"	, cat=	"TUN"	, gpu=	1	, ground=	0	, sun=	0	, video=	"techno"	, enabled=	1	}.
new Item {id=	64	, shader=	"RockyGorge"	, label=	"RockyGorge"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"canyon"	, enabled=	1	}.
new Item {id=	65	, shader=	"RollingHills"	, label=	"RollingHills"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"birds"	, enabled=	1	}.
new Item {id=	66	, shader=	"ShatteredDojo"	, label=	"ShatteredDojo"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"eerie"	, enabled=	1	}.
new Item {id=	67	, shader=	"SimplicityGalaxy"	, label=	"SimplicityGalaxy"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	68	, shader=	"SkinPeeler"	, label=	"SkinPeeler"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	69	, shader=	"SkydomeTwinSuns"	, label=	"SkydomeTwinSuns"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	70	, shader=	"SmallPlanet"	, label=	"SmallPlanet"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	71	, shader=	"StarfieldDarkDust"	, label=	"StarfieldDarkDust"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	72	, shader=	"Starnest"	, label=	"Starnest"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	73	, shader=	"StarNursery"	, label=	"StarNursery"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	74	, shader=	"Stars"	, label=	"Stars"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	1	, video=	"space"	, enabled=	1	}.
new Item {id=	75	, shader=	"Storm"	, label=	"Storm"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	76	, shader=	"SunSurface"	, label=	"SunSurface"	, cat=	"SPC"	, gpu=	1	, ground=	0	, sun=	0	, video=	"eruption"	, enabled=	1	}.
new Item {id=	77	, shader=	"TerrainPolyhedron"	, label=	"TerrainPolyhedron"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"desert"	, enabled=	1	}.
new Item {id=	78	, shader=	"Terrainz"	, label=	"Terrainz"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"mountains"	, enabled=	1	}.
new Item {id=	79	, shader=	"TinyCloud"	, label=	"TinyCloud"	, cat=	"SKY"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	80	, shader=	"TransparentIsoceles"	, label=	"TransparentIsoceles"	, cat=	"VIZ"	, gpu=	1	, ground=	0	, sun=	0	, video=	"crystal"	, enabled=	1	}.
new Item {id=	81	, shader=	"VenusBebop"	, label=	"VenusBebop"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	0	, video=	"desert"	, enabled=	1	}.
new Item {id=	82	, shader=	"Virus"	, label=	"Virus"	, cat=	"BIO"	, gpu=	1	, ground=	0	, sun=	0	, video=	"bio"	, enabled=	1	}.
new Item {id=	83	, shader=	"VolumetricLines"	, label=	"VolumetricLines"	, cat=	"MUS"	, gpu=	1	, ground=	0	, sun=	1	, video=	"techno"	, enabled=	1	}.
new Item {id=	84	, shader=	"Weather"	, label=	"Weather"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}.
new Item {id=	85	, shader=	"Xyptonjtroz"	, label=	"Xyptonjtroz"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}
};

*/
/*
public class Item
{
    public int id { get; set; }
    public string shader { get; set; }
    public string label { get; set; }
    public string cat { get; set; }
    public int gpu { get; set; }
    public float ground { get; set; }
    public int sun { get; set; }
    public string video { get; set; }
    public int enabled { get; set; }

}

public class RootObject
{
    public List<Item> items { get; set; }
}
*/
public class SkyControls : UdonSharpBehaviour
{
    [Tooltip("This list of skyboxes for the world")]
    public Material[] skybox;

    [Tooltip("This list of videos plays to match theskybox")]
    public VRCUrl[] playlist;

//    public VRCAVProVideoPlayer videoPlayer;
//    public VRCUrl  soundUrl;
    public GameObject Sun;
    public GameObject Moon;
    public bool  showSun = true;
    public bool  showMoon = false;
//    private var shaderList = new RootObject();
    public Material[] SkyboxList;
    public Text ownerTextField;
    public Text masterTextField;
    public GameObject masterCheckObj;

//    [UdonSynced]
//    VRCUrl _syncedURL;

    [UdonSynced]
    public int _syncedSky = 31;
    public int _currentSky =31;
    
    public int _newSky=0;

//    [UdonSynced]
//    int _videoNumber;
//    int _loadedVideoNumber;

//    BaseVRCVideoPlayer _currentPlayer;

    // [UdonSynced]
    // bool _ownerPlaying;
    // [UdonSynced]
    // float _videoStartNetworkTime;
    // [UdonSynced]
    // bool _ownerPaused = false;
    // bool _locallyPaused = false;

    // bool _waitForSync;
    // float _lastSyncTime;

    // [UdonSynced]
    // bool _masterOnly = true;
    // bool _masterOnlyLocal = true;
    // bool _needsOwnerTransition = false;

    // [UdonSynced]
    // int _nextPlaylistIndex = 0;

    [Tooltip("This determines whether the player will see the same sky everyone else does.")]
    public bool _globalSky = true;

//    shaderList.items = new List<Item> 
//                          {
//new Item {id=	4	, shader=	"Alps"	, label=	"Alps"	, cat=	"TER"	, gpu=	1	, ground=	0	, sun=	1	, video=	"wind"	, enabled=	1	}
//                          }

/*
        [Tooltip("If enabled defaults to unlocked so anyone can control sky")]
        public bool defaultUnlocked = false;
        
        [Tooltip("How often the sky controller should check if it is more than Sync Threshold out of sync with the time")]
        public float syncFrequency = 5.0f;
        [Tooltip("How many seconds desynced from the owner the client needs to be to trigger a resync")]
        public float syncThreshold = 0.5f;

        public GameObject masterLockedIcon;
        public Graphic lockGraphic;
        public GameObject masterUnlockedIcon;

        // Info panel elements
        public Text masterTextField;
        public Text videoOwnerTextField;
        public InputField currentSkyField;
        public InputField lastSkyField;
*/
        public GameObject lockButton;
        


    void Start()
    {
#if !UNITY_EDITOR // Causes null ref exceptions so just exclude it from the editor
        masterTextField.text = "Master: "+ Networking.GetOwner(masterCheckObj).displayName;
#endif   
    
#if !UNITY_EDITOR // Causes null ref exceptions so just exclude it from the editor
        ownerTextField.text = "Owner: " + Networking.GetOwner(gameObject).displayName;
#endif
 
    }
/*
            public void TriggerLockButton()
        {
            if (!Networking.IsMaster)
                return;

            _masterOnly = _masterOnlyLocal = !_masterOnlyLocal;

            if (_masterOnly && !Networking.IsOwner(gameObject))
            {
                _needsOwnerTransition = true;
                Networking.SetOwner(Networking.LocalPlayer, gameObject);
            }

            masterLockedIcon.SetActive(_masterOnly);
            masterUnlockedIcon.SetActive(!_masterOnly);
        }

                // Stop video button
        void StopVideo()
        {
            if (!Networking.IsOwner(gameObject))
                return;

            _videoStartNetworkTime = 0f;
            _ownerPlaying = false;
            _currentPlayer.Stop();
            _syncedURL = VRCUrl.Empty;
            _locallyPaused = _ownerPaused = false;
            _draggingSlider = false;
            _videoTargetStartTime = 0f;
        }

        */



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

    public void NextSky() 
    {
        Debug.Log("******************  NextSky Triggered  ***********************");

       _currentSky++;
       if (_currentSky>=skybox.Length) _currentSky=0;

       RenderSettings.skybox = skybox[_currentSky];
    }       
/*
        public void NextSky()
        {
            Debug.Log("NextSky event triggered");
            _currentSky++;
            if (_currentSky >= skybox.Length) _currentSky = 0;
            RenderSettings.skybox = skybox[_syncedSky];
            _syncedSky = _currentSky;
        }
*/
        public void PreviousSky()
        {
            Debug.Log("PreviousSky event triggered");
            _currentSky--;
            if (_currentSky <= -1) _currentSky = skybox.Length-1;
            RenderSettings.skybox = skybox[_syncedSky];
            _syncedSky = _currentSky;
        }

        void Update() {
            if (_newSky != _syncedSky) 
            {
                _newSky = _syncedSky;
                RenderSettings.skybox = skybox[_newSky];
            }
            if (_currentSky != _syncedSky) 
            {
                _currentSky = _syncedSky;
                RenderSettings.skybox = skybox[_syncedSky];
            }
        }

        private void showSky(int id) 
        {

            // If Global
//            if (_globalSky) 
//            {
                // If Locked by another owner
//                    if (!Networking.IsOwner(gameObject))
//                        return;
                RenderSettings.skybox = skybox[id];
                _syncedSky = id;
                _currentSky = id;

            bool isOwner = false;
#if !UNITY_EDITOR // Causes null ref exceptions so just exclude it from the editor
                isOwner = Networking.IsOwner(gameObject);
#endif

                if (!isOwner)
                {
                    Networking.SetOwner(Networking.LocalPlayer, gameObject);
                }

            // }   
            // else
            // {
            //     RenderSettings.skybox = skybox[id];
            // } 

//                if (isOwner && _needsOwnerTransition)
//                {
//                    //StopVideo();
//                    _needsOwnerTransition = false;
//                    _masterOnly = _masterOnlyLocal;
//                }


//            Networking.SetOwner(player, gameObject);


            //_ownerPlaying = false;
            //_currentPlayer.Stop();
            //_syncedURL = VRCUrl.Empty;
            //_locallyPaused = _ownerPaused = false;
            //_draggingSlider = false;
            //_videoTargetStartTime = 0f;


        }

       public void ButtonLock()
        {
            // if (!Networking.IsMaster)
            //     return;

            // _masterOnly = _masterOnlyLocal = !_masterOnlyLocal;

            // if (_masterOnly && !Networking.IsOwner(gameObject))
            // {
            //     _needsOwnerTransition = true;
            //     Networking.SetOwner(Networking.LocalPlayer, gameObject);
            // }

//            masterLockedIcon.SetActive(_masterOnly);
//            masterUnlockedIcon.SetActive(!_masterOnly);
        }

}
