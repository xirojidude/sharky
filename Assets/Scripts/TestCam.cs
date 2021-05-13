
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class TestCam : UdonSharpBehaviour
{
    public Transform FloorCollider;
    private Camera camera;
    private Transform cameraPos;
    bool grab;
    private VRCPlayerApi player;
    public RenderTexture cameraTexture;
    public Texture2D targetTexture2D;
    public Rect sourceRect;
    public float YOffset =0f;
    public float YScale = 1f;
    private Material skybox;
    public float platformThickness = 1.0f;
    public float cameraHeight = 2.0f;
 
    void Start()
    {
        camera = GetComponent<Camera>();
        cameraPos = GetComponent<Transform>();
        grab = true;
        player = Networking.LocalPlayer;

    }

    void Update()
    {
        //Quaternion headDirection =player.GetRotation();
        Quaternion headDirection = player.GetBoneRotation( HumanBodyBones.Head );
        Vector3 bodyPosition = player.GetPosition();
        cameraPos.position = bodyPosition + new Vector3(0,cameraHeight,0); // Place camera 2m above feet
        cameraPos.rotation = headDirection;
    }

    public void OnPostRender()
    {
        //sourceRect = new Rect(0, 0, RenderTexture.width, RenderTexture.height);
        int x = 0; //Mathf.FloorToInt(sourceRect.x);
        int y = 0; //Mathf.FloorToInt(sourceRect.y);
        int width = 256; //Mathf.FloorToInt(sourceRect.width);
        int height =256; // Mathf.FloorToInt(sourceRect.height);


        targetTexture2D.ReadPixels(new Rect(0, 0, 256, 256), 0, 0, false);
        targetTexture2D.Apply();
        Color[] pix = targetTexture2D.GetPixels(x, y, width, height);

        skybox = RenderSettings.skybox;
        Vector3 XYZOffset = new Vector3(0f,0f,0f);
        float Scale = 1.0f;

        if (skybox!=null) {
            XYZOffset = skybox.GetVector("_XYZPos");
            Scale = skybox.GetFloat("_Scale");
        }

        float groundDistance = (pix[0].r*256f) + (pix[0].g * 65536f) - 0.55f;


        float elevation =  -20f; //groundDistance*YScale - YOffset ;

        if (groundDistance < 255) 
        {
            if (groundDistance > 0) // ground is in sight (within 255cm of camera) 
            {
                if (groundDistance < 195) // starting to go underground - move collider up
                {
                    // calc elevation (assume we're not falling)
                    elevation = cameraPos.position.y -cameraHeight + platformThickness*.5f - groundDistance*.01f; 

                } 
                else // we are hovering above the ground - move collider down
                {
                    if (groundDistance > 205) {
                        elevation = cameraPos.position.y -cameraHeight + platformThickness*.5f - groundDistance*.01f; 

                    }
                    // otherwise we're in the deadzone 
                }
            }
            else // we are underground :(
            {
                // Are we falling ?
                // teleport player?? we can be underground by .5m before we can't tell how deep we are
            }
        }

        // Camera height above ground is [0..255] units
        // Camera should be 1m above ground
        // need to calculate scaleFactor to convert between units and meters     HeightInMeters = HeightInUnits * YScale
        // Put camera 2 meters above player's feet

        // Unity _WorldSpaceCameraPos * Scale + _XYZPos
        //  _UnityOrigin
        //  _ShaderOrigin
        //  _WorldSpaceCameraPos - get from unity
        //  _Scale - Get from material parameters??? ((Material.GetFloat("_Scale")) ... or calculate by ???
        //  _XYZPos - Get from material parameters??? (Material.GetVector("_XYZPos") ) ... or calculate
        //  _DistanceToGround - Stored as color of bottom-left pixel [0..1] (use cm as units??) 

        // Elevation = _WorldSpaceCameraPos - _DistanceToGround 

        // If ground is beyond view move collider to default level -100m 
        // otherwise 
        //      if ground is outside deadzone 2m below camera +/- 5cm
        //          recalc 
        //      otherwise  
        //          don't move the collider


        Debug.Log("Elevation:" + elevation + " Ground:" + groundDistance + " Color:"+ pix[0]);
        if (FloorCollider != null) 
        {
            // Actual height of ground is unknown because
            //  - shader XYZOffset 
            //  - distance units in shader are not consistent

            // Make collider height = ground elevation
            // ground elevation(meters) = cameraWorldHeight(meters) - cameraShaderHeight(units)
            FloorCollider.position = new Vector3(FloorCollider.position.x, elevation ,FloorCollider.position.z);

        }

        // Makes no sense ... where is ReadPixels getting a ref to the source (cameraTexture) to copy from???
        // Update: ReadPixels copies from the screen (camera's render texture) so that's why no SOURCE is specified
//        RenderTexture.active = cameraTexture;
        //RenderTexture cameraTexture = camera.activeTexture;
//        targetTexture2D.ReadPixels(new Rect(0, 0, cameraTexture.width, cameraTexture.height), 0, 0);
//        targetTexture2D.Apply();

        //Color[] pix = cameraTexture.GetPixels(x, y, width, height);
        //targetTexture2D.ReadPixels(new Rect(0, 0, cameraTexture.width, cameraTexture.height), 0, 0);
        
        //Color[] pix;
        //Texture2D targetTexture2D = new Texture2D(width, height);
        //targetTexture2D.SetPixels(pix);
        //targetTexture2D.Apply();
 
    }
}

/*
yield return new WaitForEndOfFrame();
        Rect rect = new Rect(0, 0, Screen.width, Screen.height);
        // 先创建一个的空纹理，大小可根据实现需要来设置
        Texture2D screenShot = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);

        // 读取屏幕像素信息并存储为纹理数据，
        screenShot.ReadPixels(rect, 0, 0);
        screenShot.Apply();

        // 然后将这些纹理数据，成一个png图片文件
        byte[] bytes = screenShot.EncodeToPNG();

*/
/*
WaitForEndOfFrame frameEnd = new WaitForEndOfFrame();

public IEnumerator TakeSnapshot(int width, int height)
{
    yield return frameEnd;

    Texture2D texture = new Texture2D(800, 800, TextureFormat.RGB24, true);
    texture.ReadPixels(new Rect(0, 0, 800, 800), 0, 0);
    texture.LoadRawTextureData(texture.GetRawTextureData());
    texture.Apply();
    sendTexture(texture, messageToSend);

    // gameObject.renderer.material.mainTexture = TakeSnapshot;
}
*/

/*
bool isScreenShot = false;
    private void OnPostRender()
    {
        if (isScreenShot)
        {
            Rect rect = new Rect(0, 0, Screen.width, Screen.height);
            // Create an empty texture first, and the size can be set according to the implementation needs
            Texture2D screenShot = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);

            // Read screen pixel information and store it as texture data.
            screenShot.ReadPixels(rect, 0, 0);
            screenShot.Apply();

            // The texture data is then turned into a png picture file
            byte[] bytes = screenShot.EncodeToPNG();
            print("isScreenShot "+ bytes.Length);
            isScreenShot = false;
        }
    }
    */


/*
using UnityEngine;

// A script that when attached to the camera, makes the resulting
// colors inverted. See its effect in play mode.
public class ExampleClass : MonoBehaviour
{
    private Material mat;

    // Will be called from camera after regular rendering is done.
    public void OnPostRender()
    {
        if (!mat)
        {
            // Unity has a built-in shader that is useful for drawing
            // simple colored things. In this case, we just want to use
            // a blend mode that inverts destination colors.
            var shader = Shader.Find("Hidden/Internal-Colored");
            mat = new Material(shader);
            mat.hideFlags = HideFlags.HideAndDontSave;
            // Set blend mode to invert destination colors.
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusDstColor);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
            // Turn off backface culling, depth writes, depth test.
            mat.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            mat.SetInt("_ZWrite", 0);
            mat.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);
        }

        GL.PushMatrix();
        GL.LoadOrtho();

        // activate the first shader pass (in this case we know it is the only pass)
        mat.SetPass(0);
        // draw a quad over whole screen
        GL.Begin(GL.QUADS);
        GL.Vertex3(0, 0, 0);
        GL.Vertex3(1, 0, 0);
        GL.Vertex3(1, 1, 0);
        GL.Vertex3(0, 1, 0);
        GL.End();

        GL.PopMatrix();
    }
}

*/
