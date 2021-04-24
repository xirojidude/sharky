
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class MainCamera : UdonSharpBehaviour
{
    public Transform FloorCollider;
    private Camera camera;
    bool grab;
    private VRCPlayerApi player;
    RenderTexture MyLowResTexture;
    //Texture2D deploy;

    void Start()
    {
        camera = GetComponent<Camera>();
        grab = true;
        player = Networking.LocalPlayer;
      //  deploy = new Texture2D(1,1);
      //  MyLowResTexture  = new RenderTexture(1,1,0); //deploy and MyLowResTexture should have the same resolution

    }

    void Update() 
    {
        //    Vector3 pos =  this.transform.position;
        //    pos.y = FloorCollider.position.y;
        //    FloorCollider.position = pos;
        //  FloorCollider.Translate(Vector3.up*.1f);
        // camera.Render();
        //Debug.Log("camera update");
    }

    private void OnRender()
    {

        //Blit MovieFrame to MyLowResTexture here
 
        // RenderTexture.active = MyLowResTexture ;
        // deploy.ReadPixels(new Rect(0,0,1,1), 0, 0);
        // deploy.Apply();
        
        // Color MyAverageColor = deploy.GetPixel(0,0);

    }
/*

New Texture2D, new Rect, RenderTexture active, etc., so I pushed it, but is there any other way?

Render texture → camera → in quad → captured by the camera and reflected in the coordinates of the collider's object

I'm taking this with depth, 
    but if I try to spit coordinate information out of Shader, 
    I don't want Depth 
    (that's the way to enter the player's coordinate information in the material properties 
        and put out the corresponding coordinate information as some color.
        Because it is tessellation, there is a polygon entity (alcause it is written in shader)

How to getPixel for RenderTexture 
    1.Show RenderTexture in Quad 
    2.Prepare a RenderTexture (separate from the RenderTexture in step 1??) 
        and Texture camera (rewritable dedicated texture) of the resolution you want 
    3. Shoot Quad 
    4. Stick it to the camera, 
        Create a property to receive ↑' Texture
    5.Create a function of void OnPostRender in UdonSharp 
        and process ↓(tex is ↑ Texture, cam is my own camera) tex. 
        ReadPixels(cam.pixelRect;, 0, 0, false); 
        tex.Apply(false); (This will write the camera image to Texture.) 
    6. The rest is update or something like tex. 
    GetPixel(0,0); 
    
    .. Or you can...

The setting of the texture to be read worked if it was like this, 
    and if Format is not left as RGBA32bit, 
    it will be an error I wonder if I can really do new Texture2D or something like this.
    TextureType             Default
    TextureShape            2D

    sRGB Color Texture      false
    Alpha Source            InputTextureAlpha
    Alpha is Transparency   false

    Advanced
        nonPowerOf2         None
        ReadWrite Enabled   true
        Streaming Mipmaps   false
        Generate Mipmaps    false

    Wrap Mode               Clamp
    Filter Mode             Point no filter
    
    Max Size                2048
    Resize Algorithm        Bilinear
    Format                  RGBA 32 bit
    

If you just want to use the output of the camera, you can go 1 and 3.

If you can read color information to texture at the timing of writing to RenderTexture with a camera other than a camera (such as VideoPlayer), 
    it seems to be easy to use, but I haven't examined it that much


The floor movement is like this, the floor is moving, it is a little off because it is delayed by about 2 frames (Collider felt that only one of the feet was good.

*/

    private void OnPostRender()
    {
        // Debug.Log("onPostRender");
        //  FloorCollider.Translate(Vector3.up*.1f);
        // if (grab)
        // {
        //    Camera camera = Camera.main;
        //    camera.targetTexture = renderTexture;
        //    camera.Render();
            // RenderTexture cameraTex = camera.activeTexture;
            // Vector3 pos =  player.GetPosition(); //this.transform.position;
            // pos.y = FloorCollider.position.y;
            // FloorCollider.position = pos;

            // int x = Mathf.FloorToInt(sourceRect.x);
            // int y = Mathf.FloorToInt(sourceRect.y);
            // int width = Mathf.FloorToInt(sourceRect.width);
            // int height = Mathf.FloorToInt(sourceRect.height);


            // Color[] pix = sourceTex.GetPixels(x, y, width, height);

            // FloorCollider.Translate(new Vector3(0f,.01f,0f)); // .ssetGround(1.0f);
//            Texture2D destTex = new Texture2D(width, height);
//            destTex.SetPixels(pix);
//            destTex.Apply();
            /*
            var m_Texture2D = new Texture2D(2, 2, TextureFormat.RGBA32, true);
            var mip0Data = m_Texture2D.GetPixelData<Color32>(1);


            // pixels in mip = 1 are filled with white color
            for (int i = 0; i < mip0Data.Length; i++)
            {
    //            mip0Data[i] = new Color32(255, 255, 255, 255);
            }

    //        m_Texture2D.Apply(false);
    */
        // }

    }

}
