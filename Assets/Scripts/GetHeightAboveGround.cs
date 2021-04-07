
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class GetHeightAboveGround : UdonSharpBehaviour
{
    void Start()
    {
        
    }
}
/*
private Texture2D tex;
public Color center;

void Awake()
{
    StartCoroutine(GetCenterPixel());
}


private void CreateTexture()
{
    tex = new Texture2D(1, 1, TextureFormat.RGB24, false);
}

private IEnumerator GetCenterPixel()
{
    CreateTexture();
    while (true)
    {
        yield return new WaitForEndOfFrame();
        tex.ReadPixels(new Rect(Screen.width / 2f, Screen.height / 2f, 1, 1), 0, 0);
        tex.Apply();
        center = tex.GetPixel(0,0);
    }
}
*/