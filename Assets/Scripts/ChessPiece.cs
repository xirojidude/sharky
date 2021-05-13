
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class ChessPiece : UdonSharpBehaviour
{
    public GameObject Highlight;
    private bool dropped = false;
    private bool holding = false;

    void Start()
    {
        //Piece = GetComponent<Transform>();
    }

    void FixedUpdate()
    {
        if (holding) 
        {
            float row = Mathf.Floor(this.transform.localPosition.z + .5f); 
            float col = Mathf.Floor(this.transform.localPosition.x + .5f);
            this.transform.localPosition = new Vector3(col,1f,row);
            if (Highlight!=null) 
            {
                Highlight.transform.localPosition = new Vector3(col,0f,row);
            }
        }
        if (dropped) {
            float row = Mathf.Floor(this.transform.localPosition.z + .5f); 
            float col = Mathf.Floor(this.transform.localPosition.x + .5f);

            this.transform.localPosition = new Vector3(col,0f,row);
            this.transform.localRotation = new Quaternion(0f,0f,0f,0f);
            dropped = false;
        }
    }

    void OnPickup()
    {
        float row = Mathf.Floor(this.transform.localPosition.z + .5f); 
        float col = Mathf.Floor(this.transform.localPosition.x + .5f);
        this.transform.localPosition = new Vector3(col,1f,row);
        holding = true;
        if (Highlight!=null) 
        {
            Highlight.SetActive(true);
        }
    }

    void OnDrop()
    {
        this.transform.localRotation = new Quaternion(0f,0f,0f,0f);
        holding = false;
        dropped = true;
        if (Highlight!=null) 
        {
            Highlight.SetActive(false);
        }
    }
}
