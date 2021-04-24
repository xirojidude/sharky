
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class TestCam : UdonSharpBehaviour
{
    public Transform FloorCollider;
    private Camera camera;
    bool grab;
    private VRCPlayerApi player;
    public RenderTexture cameraTexture;
    public Texture2D targetTexture2D;
    public Rect sourceRect;

    void Start()
    {
        camera = GetComponent<Camera>();
        grab = true;
        player = Networking.LocalPlayer;

    }

    void onPostRender()
    {
        //sourceRect = new Rect(0, 0, RenderTexture.width, RenderTexture.height);
        int x = 0; //Mathf.FloorToInt(sourceRect.x);
        int y = 0; //Mathf.FloorToInt(sourceRect.y);
        int width = 256; //Mathf.FloorToInt(sourceRect.width);
        int height =256; // Mathf.FloorToInt(sourceRect.height);


        Color[] pix = targetTexture2D.GetPixels(x, y, width, height);

        // Makes no sense ... where is ReadPixels getting a ref to the source (cameraTexture) to copy from???
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
